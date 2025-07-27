// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::Borrow;
use std::cell::UnsafeCell;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::FusedIterator;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::slice::{Iter as SliceIter, IterMut as SliceIterMut};
use std::{mem, ptr};

use archery::{SharedPointer, SharedPointerKind};
use bitmaps::{Bits, BitsImpl};
use imbl_sized_chunks::inline_array::InlineArray;
use imbl_sized_chunks::sparse_chunk::{Iter as ChunkIter, IterMut as ChunkIterMut, SparseChunk};

use crate::config::HASH_LEVEL_SIZE as HASH_SHIFT;
pub(crate) type HashBits = <BitsImpl<HASH_WIDTH> as Bits>::Store; // a uint of HASH_WIDTH bits

const HASH_WIDTH: usize = 2_usize.pow(HASH_SHIFT as u32);
const HASH_MASK: HashBits = (HASH_WIDTH - 1) as HashBits;
const ITER_STACK_CAPACITY: usize = (HASH_WIDTH / HASH_SHIFT) + 1;
const SMALL_NODE_WIDTH: usize = HASH_WIDTH / 2;
const SMALL_NODE_MASK: HashBits = (SMALL_NODE_WIDTH - 1) as HashBits;

pub(crate) fn hash_key<K: Hash + ?Sized, S: BuildHasher>(bh: &S, key: &K) -> HashBits {
    let mut hasher = bh.build_hasher();
    key.hash(&mut hasher);
    hasher.finish() as HashBits
}

#[inline]
fn mask(hash: HashBits, shift: usize) -> HashBits {
    hash >> shift & HASH_MASK
}

pub trait HashValue {
    type Key: Eq;

    fn extract_key(&self) -> &Self::Key;
    fn ptr_eq(&self, other: &Self) -> bool;
}

pub(crate) struct Node<A, P: SharedPointerKind> {
    /// Whether this node is using linear probing for collision resolution.
    /// When true all child nodes are `Value`s.
    linear_probing: bool,
    data: SparseChunk<Entry<A, P>, HASH_WIDTH>,
}

impl<A: Clone, P: SharedPointerKind> Clone for Node<A, P> {
    fn clone(&self) -> Self {
        Self {
            linear_probing: self.linear_probing.clone(),
            data: self.data.clone(),
        }
    }
}

pub(crate) struct SmallNode<A, P: SharedPointerKind> {
    data: SparseChunk<Entry<A, P>, SMALL_NODE_WIDTH>,
}

impl<A: Clone, P: SharedPointerKind> Clone for SmallNode<A, P> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<A, P: SharedPointerKind> Default for SmallNode<A, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A, P: SharedPointerKind> SmallNode<A, P> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        SmallNode {
            data: SparseChunk::new(),
        }
    }

    #[inline]
    fn with(with: impl FnOnce(&mut Self)) -> SharedPointer<Self, P> {
        let result: SharedPointer<UnsafeCell<mem::MaybeUninit<Self>>, P> =
            SharedPointer::new(UnsafeCell::new(mem::MaybeUninit::uninit()));
        #[allow(unsafe_code)]
        unsafe {
            (&mut *result.get()).write(SmallNode::new());
            let mut_ptr = &mut *UnsafeCell::raw_get(&*result);
            let mut_ptr = MaybeUninit::as_mut_ptr(mut_ptr);
            with(&mut *mut_ptr);
            let result = ManuallyDrop::new(result);
            mem::transmute_copy(&result)
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn mask(hash: HashBits, shift: usize) -> HashBits {
        hash >> shift & SMALL_NODE_MASK
    }

    fn pop(&mut self) -> Entry<A, P> {
        self.data.pop().unwrap()
    }
}

impl<A: HashValue, P: SharedPointerKind> SmallNode<A, P> {
    pub(crate) fn get<BK>(&self, hash: HashBits, shift: usize, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = Self::mask(hash, shift) as usize;
        match self.data.get(index) {
            Some(Entry::Value(ref value, value_hash)) => {
                if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                    Some(value)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub(crate) fn get_mut<BK>(&mut self, hash: HashBits, shift: usize, key: &BK) -> Option<&mut A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = Self::mask(hash, shift) as usize;
        match self.data.get_mut(index) {
            Some(Entry::Value(ref mut value, value_hash)) => {
                if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                    Some(value)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub(crate) fn insert(&mut self, hash: HashBits, shift: usize, value: A) -> Result<Option<A>, A>
    where
        A: Clone,
    {
        let index = Self::mask(hash, shift) as usize;
        match self.data.get_mut(index) {
            Some(Entry::Value(ref mut existing, existing_hash)) => {
                if hash_may_eq::<A>(hash, *existing_hash)
                    && existing.extract_key() == value.extract_key()
                {
                    Ok(Some(mem::replace(existing, value)))
                } else {
                    Err(value)
                }
            }
            None => {
                self.data.insert(index, Entry::Value(value, hash));
                Ok(None)
            }
            _ => unreachable!("SmallNode should only contain Values"),
        }
    }

    pub(crate) fn remove<BK>(&mut self, hash: HashBits, shift: usize, key: &BK) -> Option<A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = Self::mask(hash, shift) as usize;
        match self.data.get(index) {
            Some(Entry::Value(value, value_hash)) => {
                if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                    self.data.remove(index).map(Entry::unwrap_value)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct CollisionNode<A> {
    hash: HashBits,
    data: Vec<A>,
}

pub(crate) enum Entry<A, P: SharedPointerKind> {
    Value(A, HashBits),
    Collision(SharedPointer<CollisionNode<A>, P>),
    Node(SharedPointer<Node<A, P>, P>),
    SmallNode(SharedPointer<SmallNode<A, P>, P>),
}

impl<A: Clone, P: SharedPointerKind> Clone for Entry<A, P> {
    fn clone(&self) -> Self {
        match self {
            Entry::Value(value, hash) => Entry::Value(value.clone(), *hash),
            Entry::Collision(coll) => Entry::Collision(coll.clone()),
            Entry::Node(node) => Entry::Node(node.clone()),
            Entry::SmallNode(node) => Entry::SmallNode(node.clone()),
        }
    }
}

impl<A, P: SharedPointerKind> Entry<A, P> {
    fn is_value(&self) -> bool {
        matches!(self, Entry::Value(_, _))
    }

    fn unwrap_value(self) -> A {
        match self {
            Entry::Value(a, _) => a,
            _ => panic!("nodes::hamt::Entry::unwrap_value: unwrapped a non-value"),
        }
    }
}

impl<A, P: SharedPointerKind> From<CollisionNode<A>> for Entry<A, P> {
    fn from(node: CollisionNode<A>) -> Self {
        Entry::Collision(SharedPointer::new(node))
    }
}

impl<A, P: SharedPointerKind> Default for Node<A, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A, P: SharedPointerKind> Node<A, P> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Node {
            linear_probing: true,
            data: SparseChunk::new(),
        }
    }

    /// Special constructor to allow initializing Nodes w/o incurring multiple memory copies.
    /// These copies really slow things down once Node crosses a certain size threshold and copies become calls to memcopy.
    #[inline]
    fn with(with: impl FnOnce(&mut Self)) -> SharedPointer<Self, P> {
        let result: SharedPointer<UnsafeCell<mem::MaybeUninit<Self>>, P> =
            SharedPointer::new(UnsafeCell::new(mem::MaybeUninit::uninit()));
        #[allow(unsafe_code)]
        unsafe {
            // Initialize the MaybeUninit node
            (&mut *result.get()).write(Node::new());
            // Dereference all the way to the newly initialized &mut Node
            let mut_ptr = &mut *UnsafeCell::raw_get(&*result);
            let mut_ptr = MaybeUninit::as_mut_ptr(mut_ptr);
            with(&mut *mut_ptr);
            // Note that transmute isn't usable with the generic argument P, so we have to use
            // a combination of transmute_copy and ManuallyDrop
            // Safety: UnsafeCell<_> and UnsafeCell<MaybeUninit<_>> have the same memory representation
            //         and ManuallyDrop<T> and T have the same memory representation
            let result = ManuallyDrop::new(result);
            mem::transmute_copy(&result)
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn pair(
        index1: usize,
        value1: Entry<A, P>,
        mut index2: usize,
        value2: Entry<A, P>,
    ) -> SharedPointer<Self, P> {
        if index1 == index2 {
            index2 = (index1 + 1) % HASH_WIDTH;
        }
        Self::with(|this| {
            this.data.insert(index1, value1);
            this.data.insert(index2, value2);
        })
    }

    fn pop(&mut self) -> Entry<A, P> {
        self.data.pop().unwrap()
    }
}

impl<A: HashValue, P: SharedPointerKind> Node<A, P> {
    #[inline]
    fn merge_values(
        value1: A,
        hash1: HashBits,
        value2: A,
        hash2: HashBits,
        shift: usize,
    ) -> Entry<A, P> {
        let small_index1 = SmallNode::<A, P>::mask(hash1, shift) as usize;
        let small_index2 = SmallNode::<A, P>::mask(hash2, shift) as usize;

        if small_index1 != small_index2 {
            // Safe to use SmallNode - no information loss
            let small_node = SmallNode::with(|node| {
                node.data.insert(small_index1, Entry::Value(value1, hash1));
                node.data.insert(small_index2, Entry::Value(value2, hash2));
            });
            Entry::SmallNode(small_node)
        } else {
            // Would collide in SmallNode, create regular Node
            let node = Node::pair(
                mask(hash1, shift) as usize,
                Entry::Value(value1, hash1),
                mask(hash2, shift) as usize,
                Entry::Value(value2, hash2),
            );
            Entry::Node(node)
        }
    }

    pub(crate) fn get<BK>(&self, hash: HashBits, shift: usize, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let mut index = mask(hash, shift) as usize;
        while let Some(entry) = self.data.get(index) {
            return match entry {
                Entry::Value(ref value, value_hash) => {
                    if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                        Some(value)
                    } else if !self.linear_probing {
                        None
                    } else {
                        index = (index + 1) % HASH_WIDTH;
                        continue;
                    }
                }
                Entry::Collision(ref coll) => coll.get(key),
                Entry::Node(ref child) => child.get(hash, shift + HASH_SHIFT, key),
                Entry::SmallNode(ref small) => small.get(hash, shift + HASH_SHIFT, key),
            };
        }
        None
    }

    pub(crate) fn get_mut<BK>(&mut self, hash: HashBits, shift: usize, key: &BK) -> Option<&mut A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let this = self as *mut Self;
        #[allow(dropping_references)]
        drop(self); // prevent self from being used or moved, so it's is safe to dereference `this` later
        let mut index = mask(hash, shift) as usize;
        loop {
            // Restore a mutable reference to self to avoid hitting the borrow checker
            // limitation that prevents us from returning mutable references from the original
            // `self` inside a loop. This is safe because we only restore the mutable reference
            // once per iteration and the references goes out of scope at the end of the loop.
            #[allow(unsafe_code)]
            let this = unsafe { &mut *this };
            return match this.data.get_mut(index) {
                Some(Entry::Node(ref mut child_ref)) => {
                    SharedPointer::make_mut(child_ref).get_mut(hash, shift + HASH_SHIFT, key)
                }
                Some(Entry::SmallNode(ref mut small_ref)) => {
                    SharedPointer::make_mut(small_ref).get_mut(hash, shift + HASH_SHIFT, key)
                }
                Some(Entry::Value(ref mut value, value_hash)) => {
                    if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                        Some(value)
                    } else if !this.linear_probing {
                        None
                    } else {
                        index = (index + 1) % HASH_WIDTH;
                        continue;
                    }
                }
                Some(Entry::Collision(ref mut coll_ref)) => {
                    SharedPointer::make_mut(coll_ref).get_mut(key)
                }
                None => None,
            };
        }
    }

    #[allow(unsafe_code)]
    pub(crate) fn insert(&mut self, hash: HashBits, shift: usize, value: A) -> Option<A>
    where
        A: Clone,
    {
        let mut index = mask(hash, shift) as usize;
        while let Some(entry) = self.data.get_mut(index) {
            // Value is here
            match entry {
                // Update value or create a subtree
                Entry::Value(ref mut current, current_hash) => {
                    if hash_may_eq::<A>(hash, *current_hash)
                        && current.extract_key() == value.extract_key()
                    {
                        return Some(mem::replace(current, value));
                    }
                    if self.linear_probing {
                        index = (index + 1) % HASH_WIDTH;
                        continue;
                    }
                }
                // There's already a collision here.
                Entry::Collision(ref mut collision) => {
                    let coll = SharedPointer::make_mut(collision);
                    return coll.insert(value);
                }
                Entry::Node(ref mut child_ref) => {
                    // Child node
                    let child = SharedPointer::make_mut(child_ref);
                    return child.insert(hash, shift + HASH_SHIFT, value);
                }
                Entry::SmallNode(ref mut small_ref) => {
                    // SmallNode - first check if we need to upgrade
                    let small = SharedPointer::make_mut(small_ref);
                    match small.insert(hash, shift + HASH_SHIFT, value) {
                        Ok(result) => return result,
                        Err(value) => {
                            // It's a collision, need to upgrade to Node
                            let mut node = Node::new();
                            for entry in mem::take(&mut small.data) {
                                if let Entry::Value(v, h) = entry {
                                    node.insert(h, shift + HASH_SHIFT, v);
                                } else {
                                    unreachable!("SmallNode should only contain Values");
                                }
                            }

                            // Insert the new value
                            node.insert(hash, shift + HASH_SHIFT, value);
                            *entry = Entry::Node(SharedPointer::new(node));
                            return None;
                        }
                    }
                }
            }

            // If we get here, we're inserting a value over an exiting value (collision).
            // We're going to be unsafe and pry it out of the reference, trusting
            // that we overwrite it with the merged node.
            let Entry::Value(old_value, old_hash) = (unsafe { ptr::read(entry) }) else {
                unreachable!()
            };
            let new_entry = if shift + HASH_SHIFT >= HASH_WIDTH {
                // We're at the lowest level, need to set up a collision node.
                let coll = CollisionNode::new(hash, old_value, value);
                Entry::from(coll)
            } else {
                Node::merge_values(old_value, old_hash, value, hash, shift + HASH_SHIFT)
            };
            unsafe { ptr::write(entry, new_entry) };
            return None;
        }

        if self.linear_probing && self.data.len() >= HASH_WIDTH / 2 {
            self.linear_probing = false;
            let old_data = mem::take(&mut self.data);
            for entry in old_data.drain() {
                if let Entry::Value(value, hash) = entry {
                    self.insert(hash, shift, value);
                } else {
                    unreachable!("linear probing should only contain values")
                }
            }
            return self.insert(hash, shift, value);
        }
        self.data.insert(index, Entry::Value(value, hash));
        None
    }

    pub(crate) fn remove<BK>(&mut self, hash: HashBits, shift: usize, key: &BK) -> Option<A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let mut index = mask(hash, shift) as usize;
        // First find the entry to remove
        loop {
            match self.data.get(index) {
                None => return None,
                Some(Entry::Value(value, value_hash)) => {
                    if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                        break;
                    } else if !self.linear_probing {
                        return None;
                    } else {
                        index = (index + 1) % HASH_WIDTH;
                    }
                }
                Some(Entry::Collision(_) | Entry::Node(_) | Entry::SmallNode(_)) => break,
            }
        }

        let new_node;
        let removed;
        match self.data.get_mut(index).unwrap() {
            Entry::Node(ref mut child_ref) => {
                let child = SharedPointer::make_mut(child_ref);
                match child.remove(hash, shift + HASH_SHIFT, key) {
                    None => return None,
                    Some(value) => {
                        if child.len() == 1
                            && child.data[child.data.first_index().unwrap()].is_value()
                        {
                            removed = Some(value);
                            new_node = Some(child.pop());
                        } else {
                            return Some(value);
                        }
                    }
                }
            }
            Entry::Value(..) => {
                new_node = None;
                removed = self.data.remove(index).map(Entry::unwrap_value);
            }
            Entry::Collision(ref mut coll_ref) => {
                let coll = SharedPointer::make_mut(coll_ref);
                removed = coll.remove(key);
                if coll.len() == 1 {
                    new_node = Some(coll.pop());
                } else {
                    return removed;
                }
            }
            Entry::SmallNode(ref mut small_ref) => {
                let small = SharedPointer::make_mut(small_ref);
                match small.remove(hash, shift + HASH_SHIFT, key) {
                    None => return None,
                    Some(value) => {
                        if small.len() == 1 {
                            removed = Some(value);
                            new_node = Some(small.pop());
                        } else {
                            return Some(value);
                        }
                    }
                }
            }
        }

        if let Some(node) = new_node {
            self.data.insert(index, node);
        } else if self.linear_probing {
            // Perform backwards shift if using linear probing
            let mut next = (index + 1) % HASH_WIDTH;
            while let Some(Entry::Value(_, value_hash)) = self.data.get(next) {
                let ideal_index = mask(*value_hash, shift) as usize;
                let next_dib = next.wrapping_sub(ideal_index) % HASH_WIDTH;
                let index_dib = index.wrapping_sub(ideal_index) % HASH_WIDTH;
                if index_dib < next_dib {
                    let entry = self.data.remove(next).unwrap();
                    self.data.insert(index, entry);
                    index = next;
                }
                next = (next + 1) % HASH_WIDTH;
            }
        }

        removed
    }
}

/// Compare two hashes, returning true if they are equal or if the Key is (likely) cheap to compare.
#[inline]
fn hash_may_eq<A: HashValue>(hash: u32, other_hash: u32) -> bool {
    (!mem::needs_drop::<A::Key>() && mem::size_of::<A::Key>() <= 16) || hash == other_hash
}

impl<A: HashValue> CollisionNode<A> {
    fn new(hash: HashBits, value1: A, value2: A) -> Self {
        CollisionNode {
            hash,
            data: vec![value1, value2],
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    fn get<BK>(&self, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        self.data
            .iter()
            .find(|&entry| key == entry.extract_key().borrow())
    }

    fn get_mut<BK>(&mut self, key: &BK) -> Option<&mut A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        self.data
            .iter_mut()
            .find(|entry| key == entry.extract_key().borrow())
    }

    fn insert(&mut self, value: A) -> Option<A> {
        for item in &mut self.data {
            if value.extract_key() == item.extract_key() {
                return Some(mem::replace(item, value));
            }
        }
        self.data.push(value);
        None
    }

    fn remove<BK>(&mut self, key: &BK) -> Option<A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let mut loc = None;
        for (index, item) in self.data.iter().enumerate() {
            if key == item.extract_key().borrow() {
                loc = Some(index);
            }
        }
        if let Some(index) = loc {
            Some(self.data.remove(index))
        } else {
            None
        }
    }

    fn pop<P: SharedPointerKind>(&mut self) -> Entry<A, P> {
        Entry::Value(self.data.pop().unwrap(), self.hash)
    }
}

impl<A, P: SharedPointerKind> Node<A, P> {
    /// Analyze the node structure for debugging/statistics
    pub(crate) fn analyze_structure<F>(&self, mut visitor: F)
    where
        F: FnMut(&Entry<A, P>),
    {
        for i in self.data.indices() {
            visitor(&self.data[i]);
        }
    }
}

/// An allocation-free stack for iterators.
type InlineStack<T> = InlineArray<T, (usize, [T; ITER_STACK_CAPACITY])>;

enum IterItem<'a, A, P: SharedPointerKind> {
    Node(ChunkIter<'a, Entry<A, P>, HASH_WIDTH>),
    SmallNode(ChunkIter<'a, Entry<A, P>, SMALL_NODE_WIDTH>),
}

// We manually impl Clone for IterItem to allow cloning even when A isn't Clone
// This works because the iterators hold references, not owned values
impl<'a, A, P: SharedPointerKind> Clone for IterItem<'a, A, P> {
    fn clone(&self) -> Self {
        match self {
            IterItem::Node(iter) => IterItem::Node(iter.clone()),
            IterItem::SmallNode(iter) => IterItem::SmallNode(iter.clone()),
        }
    }
}

// Ref iterator

pub(crate) struct Iter<'a, A, P: SharedPointerKind> {
    count: usize,
    stack: InlineStack<IterItem<'a, A, P>>,
    collision: Option<(HashBits, SliceIter<'a, A>)>,
}

// We impl Clone instead of deriving it, because we want Clone even if K and V aren't.
impl<'a, A, P: SharedPointerKind> Clone for Iter<'a, A, P> {
    fn clone(&self) -> Self {
        Self {
            count: self.count,
            stack: self.stack.clone(),
            collision: self.collision.clone(),
        }
    }
}

impl<'a, A, P> Iter<'a, A, P>
where
    A: 'a,
    P: SharedPointerKind,
{
    pub(crate) fn new(root: Option<&'a Node<A, P>>, size: usize) -> Self {
        let mut result = Iter {
            count: size,
            stack: InlineStack::new(),
            collision: None,
        };
        if let Some(node) = root {
            result.stack.push(IterItem::Node(node.data.iter()));
        }
        result
    }
}

impl<'a, A, P> Iterator for Iter<'a, A, P>
where
    A: 'a,
    P: SharedPointerKind,
{
    type Item = (&'a A, HashBits);

    fn next(&mut self) -> Option<Self::Item> {
        'outer: loop {
            if let Some((hash, ref mut coll)) = self.collision {
                match coll.next() {
                    None => self.collision = None,
                    Some(value) => {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                };
            }

            while let Some(current) = self.stack.last_mut() {
                let next_entry = match current {
                    IterItem::Node(iter) => iter.next(),
                    IterItem::SmallNode(iter) => iter.next(),
                };

                match next_entry {
                    Some(Entry::Value(value, hash)) => {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                    Some(Entry::Node(child)) => {
                        self.stack.push(IterItem::Node(child.data.iter()));
                    }
                    Some(Entry::SmallNode(small)) => {
                        self.stack.push(IterItem::SmallNode(small.data.iter()));
                    }
                    Some(Entry::Collision(coll)) => {
                        self.collision = Some((coll.hash, coll.data.iter()));
                        continue 'outer;
                    }
                    None => {
                        self.stack.pop();
                    }
                }
            }
            return None;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<'a, A, P: SharedPointerKind> ExactSizeIterator for Iter<'a, A, P> where A: 'a {}

impl<'a, A, P: SharedPointerKind> FusedIterator for Iter<'a, A, P> where A: 'a {}

// Mut ref iterator

enum IterMutItem<'a, A, P: SharedPointerKind> {
    Node(ChunkIterMut<'a, Entry<A, P>, HASH_WIDTH>),
    SmallNode(ChunkIterMut<'a, Entry<A, P>, SMALL_NODE_WIDTH>),
}

pub(crate) struct IterMut<'a, A, P: SharedPointerKind> {
    count: usize,
    stack: InlineStack<IterMutItem<'a, A, P>>,
    collision: Option<(HashBits, SliceIterMut<'a, A>)>,
}

impl<'a, A, P> IterMut<'a, A, P>
where
    A: 'a,
    P: SharedPointerKind,
{
    pub(crate) fn new(root: Option<&'a mut Node<A, P>>, size: usize) -> Self {
        let mut result = IterMut {
            count: size,
            stack: InlineStack::new(),
            collision: None,
        };
        if let Some(node) = root {
            result.stack.push(IterMutItem::Node(node.data.iter_mut()));
        }
        result
    }
}

impl<'a, A, P> Iterator for IterMut<'a, A, P>
where
    A: Clone + 'a,
    P: SharedPointerKind,
{
    type Item = (&'a mut A, HashBits);

    fn next(&mut self) -> Option<Self::Item> {
        'outer: loop {
            if let Some((hash, ref mut coll)) = self.collision {
                match coll.next() {
                    None => self.collision = None,
                    Some(value) => {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                };
            }

            while let Some(current) = self.stack.last_mut() {
                let next_entry = match current {
                    IterMutItem::Node(iter) => iter.next(),
                    IterMutItem::SmallNode(iter) => iter.next(),
                };

                match next_entry {
                    Some(Entry::Value(value, hash)) => {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                    Some(Entry::Node(child_ref)) => {
                        let child = SharedPointer::make_mut(child_ref);
                        self.stack.push(IterMutItem::Node(child.data.iter_mut()));
                    }
                    Some(Entry::SmallNode(small_ref)) => {
                        let small = SharedPointer::make_mut(small_ref);
                        self.stack
                            .push(IterMutItem::SmallNode(small.data.iter_mut()));
                    }
                    Some(Entry::Collision(coll_ref)) => {
                        let coll = SharedPointer::make_mut(coll_ref);
                        self.collision = Some((coll.hash, coll.data.iter_mut()));
                        continue 'outer;
                    }
                    None => {
                        self.stack.pop();
                    }
                }
            }
            return None;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<'a, A, P: SharedPointerKind> ExactSizeIterator for IterMut<'a, A, P> where A: Clone + 'a {}

impl<'a, A, P: SharedPointerKind> FusedIterator for IterMut<'a, A, P> where A: Clone + 'a {}

// Consuming iterator

enum DrainItem<A, P: SharedPointerKind> {
    Node(SharedPointer<Node<A, P>, P>),
    SmallNode(SharedPointer<SmallNode<A, P>, P>),
    Collision(SharedPointer<CollisionNode<A>, P>),
}

pub(crate) struct Drain<A, P: SharedPointerKind> {
    count: usize,
    stack: InlineStack<DrainItem<A, P>>,
}

impl<A, P: SharedPointerKind> Drain<A, P> {
    pub(crate) fn new(root: Option<SharedPointer<Node<A, P>, P>>, size: usize) -> Self {
        let mut result = Drain {
            count: size,
            stack: InlineStack::new(),
        };
        if let Some(root) = root {
            result.stack.push(DrainItem::Node(root));
        }
        result
    }
}

impl<A, P: SharedPointerKind> Iterator for Drain<A, P>
where
    A: Clone,
{
    type Item = (A, HashBits);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(current) = self.stack.last_mut() {
            match current {
                DrainItem::Node(node_ref) => match SharedPointer::make_mut(node_ref).data.pop() {
                    Some(Entry::Value(value, hash)) => {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                    Some(Entry::Node(child)) => {
                        self.stack.push(DrainItem::Node(child));
                    }
                    Some(Entry::SmallNode(small)) => {
                        self.stack.push(DrainItem::SmallNode(small));
                    }
                    Some(Entry::Collision(coll)) => {
                        self.stack.push(DrainItem::Collision(coll));
                    }
                    None => {
                        self.stack.pop();
                    }
                },
                DrainItem::SmallNode(small_ref) => {
                    let small = SharedPointer::make_mut(small_ref);
                    if let Some(Entry::Value(value, hash)) = small.data.pop() {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                    self.stack.pop();
                }
                DrainItem::Collision(coll_ref) => {
                    let coll = SharedPointer::make_mut(coll_ref);
                    if let Some(value) = coll.data.pop() {
                        self.count -= 1;
                        return Some((value, coll.hash));
                    }
                    self.stack.pop();
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<A, P: SharedPointerKind> ExactSizeIterator for Drain<A, P> where A: Clone {}

impl<A, P: SharedPointerKind> FusedIterator for Drain<A, P> where A: Clone {}

impl<A: fmt::Debug, P: SharedPointerKind> fmt::Debug for Node<A, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Node[ ")?;
        for i in self.data.indices() {
            write!(f, "{}: ", i)?;
            match &self.data[i] {
                Entry::Value(v, h) => write!(f, "{:?} :: {}, ", v, h)?,
                Entry::Collision(c) => write!(f, "Coll{:?} :: {}", c.data, c.hash)?,
                Entry::Node(n) => write!(f, "{:?}, ", n)?,
                Entry::SmallNode(s) => write!(f, "{:?}, ", s)?,
            }
        }
        write!(f, " ]")
    }
}

impl<A: fmt::Debug, P: SharedPointerKind> fmt::Debug for SmallNode<A, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "SmallNode[ ")?;
        for i in self.data.indices() {
            write!(f, "{}: ", i)?;
            match &self.data[i] {
                Entry::Value(v, h) => write!(f, "{:?} :: {}, ", v, h)?,
                _ => unreachable!("SmallNode should only contain Values"),
            }
        }
        write!(f, " ]")
    }
}
