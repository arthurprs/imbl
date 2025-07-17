// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![feature(test)]

extern crate imbl;
extern crate rand;
extern crate rpds;
extern crate test;

use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::iter::FromIterator;
use std::sync::Arc;
use test::Bencher;

use archery::ArcTK;
use imbl::ordmap::OrdMap;
use rpds::RedBlackTreeMapSync;

// Trait to abstract over different map implementations
trait BenchMap<K, V>: Clone + FromIterator<(K, V)>
where
    K: Clone + Ord,
    V: Clone,
{
    const IMMUTABLE: bool = true;
    type Iter<'a>: Iterator<Item = (&'a K, &'a V)>
    where
        Self: 'a,
        K: 'a,
        V: 'a;
    type RangeIter<'a>: Iterator<Item = (&'a K, &'a V)>
    where
        Self: 'a,
        K: 'a,
        V: 'a;

    fn new() -> Self;
    fn insert(&mut self, k: K, v: V) -> Option<V>;
    fn insert_clone(&self, k: K, v: V) -> Self;
    fn remove(&mut self, k: &K) -> Option<V>;
    fn remove_clone(&self, k: &K) -> Self;
    fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized;
    fn iter(&self) -> Self::Iter<'_>;
    fn range<'a>(&'a self, range: std::ops::RangeFrom<&'a K>) -> Self::RangeIter<'a>;
    fn is_empty(&self) -> bool;
    fn without_min(&self) -> (Option<(K, V)>, Self);
    fn without_max(&self) -> (Option<(K, V)>, Self);
}

// Implementation for OrdMap
impl<K, V> BenchMap<K, V> for OrdMap<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    type Iter<'a>
        = imbl::ordmap::Iter<'a, K, V, imbl::shared_ptr::DefaultSharedPtr>
    where
        K: 'a,
        V: 'a;
    type RangeIter<'a>
        = imbl::ordmap::RangedIter<'a, K, V, imbl::shared_ptr::DefaultSharedPtr>
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        OrdMap::new()
    }

    fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.insert(k, v)
    }

    fn insert_clone(&self, k: K, v: V) -> Self {
        self.update(k, v)
    }

    fn remove(&mut self, k: &K) -> Option<V> {
        self.remove(k)
    }

    fn remove_clone(&self, k: &K) -> Self {
        self.without(k)
    }

    fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(k)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn range<'a>(&'a self, range: std::ops::RangeFrom<&'a K>) -> Self::RangeIter<'a> {
        self.range(range)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn without_min(&self) -> (Option<(K, V)>, Self) {
        self.without_min_with_key()
    }

    fn without_max(&self) -> (Option<(K, V)>, Self) {
        self.without_max_with_key()
    }
}

// Implementation for RedBlackTreeMapSync
impl<K, V> BenchMap<K, V> for RedBlackTreeMapSync<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    type Iter<'a>
        = rpds::map::red_black_tree_map::Iter<'a, K, V, ArcTK>
    where
        K: 'a,
        V: 'a;
    type RangeIter<'a>
        = std::iter::Map<
        rpds::map::red_black_tree_map::RangeIter<'a, K, V, std::ops::RangeFrom<&'a K>, K, ArcTK>,
        fn((&'a K, &'a V)) -> (&'a K, &'a V),
    >
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        RedBlackTreeMapSync::new_sync()
    }

    fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.insert_mut(k, v);
        None
    }

    fn insert_clone(&self, k: K, v: V) -> Self {
        self.insert(k, v)
    }

    fn remove(&mut self, k: &K) -> Option<V> {
        if self.remove_mut(k) {
            None // rpds doesn't return the removed value
        } else {
            None
        }
    }

    fn remove_clone(&self, k: &K) -> Self {
        self.remove(k)
    }

    fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(k)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn range<'a>(&'a self, range: std::ops::RangeFrom<&'a K>) -> Self::RangeIter<'a> {
        self.range::<K, _>(range).map(|(k, v)| (k, v))
    }

    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    fn without_min(&self) -> (Option<(K, V)>, Self) {
        match self.first() {
            Some((k, _)) => {
                let k = k.clone();
                let new_map = self.remove(&k);
                (self.get(&k).map(|v| (k, v.clone())), new_map)
            }
            None => (None, self.clone()),
        }
    }

    fn without_max(&self) -> (Option<(K, V)>, Self) {
        match self.last() {
            Some((k, _)) => {
                let k = k.clone();
                let new_map = self.remove(&k);
                (self.get(&k).map(|v| (k, v.clone())), new_map)
            }
            None => (None, self.clone()),
        }
    }
}

// Implementation for BTreeMap
impl<K, V> BenchMap<K, V> for BTreeMap<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    const IMMUTABLE: bool = false;
    type Iter<'a>
        = std::collections::btree_map::Iter<'a, K, V>
    where
        K: 'a,
        V: 'a;
    type RangeIter<'a>
        = std::collections::btree_map::Range<'a, K, V>
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        BTreeMap::new()
    }

    fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.insert(k, v)
    }

    fn insert_clone(&self, _k: K, _v: V) -> Self {
        Self::new()
    }

    fn remove(&mut self, k: &K) -> Option<V> {
        self.remove(k)
    }

    fn remove_clone(&self, _k: &K) -> Self {
        Self::new()
    }

    fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(k)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn range<'a>(&'a self, range: std::ops::RangeFrom<&'a K>) -> Self::RangeIter<'a> {
        self.range(range)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn without_min(&self) -> (Option<(K, V)>, Self) {
        unreachable!()
    }

    fn without_max(&self) -> (Option<(K, V)>, Self) {
        unreachable!()
    }
}

// Trait for generating test data
trait TestData: Clone + Ord + Debug {
    fn generate(size: usize) -> Vec<Self>;
}

impl TestData for i64 {
    fn generate(size: usize) -> Vec<Self> {
        let mut gen = SmallRng::seed_from_u64(1);
        let mut set = BTreeSet::new();
        while set.len() < size {
            let next = gen.random::<i64>();
            set.insert(next);
        }
        set.into_iter().collect()
    }
}
impl TestData for String {
    fn generate(size: usize) -> Vec<Self> {
        let mut gen = SmallRng::seed_from_u64(1);
        let mut set = BTreeSet::new();
        while set.len() < size {
            let len = gen.random_range(5..20);
            let s: String = (0..len)
                .map(|_| gen.random_range(b'a'..=b'z') as char)
                .collect();
            set.insert(s);
        }
        set.into_iter().collect()
    }
}

impl<T> TestData for Arc<T>
where
    T: TestData + 'static,
{
    fn generate(size: usize) -> Vec<Self> {
        T::generate(size).into_iter().map(Arc::new).collect()
    }
}

fn reorder<A: Clone>(vec: &[A]) -> Vec<A> {
    let mut gen = SmallRng::seed_from_u64(1);
    let mut out = vec.to_vec();
    out.shuffle(&mut gen);
    out
}

// Generic benchmark functions
fn bench_lookup<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let order = reorder(&keys);
    let m: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        for k in &order {
            let _ = m.get(k);
        }
    })
}

fn bench_insert<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    if !M::IMMUTABLE {
        return; // Skip for non-immutable maps
    }
    let keys = K::generate(size);
    let values = V::generate(size);
    b.iter(|| {
        let mut m = M::new();
        for (k, v) in keys.clone().into_iter().zip(values.clone()) {
            m = m.insert_clone(k, v);
        }
    })
}

fn bench_insert_mut<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    b.iter(|| {
        let mut m = M::new();
        for (k, v) in keys.clone().into_iter().zip(values.clone()) {
            m.insert(k, v);
        }
    })
}

fn bench_remove<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    if !M::IMMUTABLE {
        return; // Skip for non-immutable maps
    }
    let keys = K::generate(size);
    let values = V::generate(size);
    let order = reorder(&keys);
    let map: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        let mut m = map.clone();
        for k in &order {
            m = m.remove_clone(k);
        }
    })
}

fn bench_remove_mut<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let order = reorder(&keys);
    let map: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        let mut m = map.clone();
        for k in &order {
            m.remove(k);
        }
    })
}

fn bench_remove_min<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    if !M::IMMUTABLE {
        return; // Skip for non-immutable maps
    }
    let keys = K::generate(size);
    let values = V::generate(size);
    let map: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        let mut m = map.clone();
        assert!(!m.is_empty());
        for _ in 0..size {
            m = m.without_min().1;
        }
        assert!(m.is_empty())
    })
}

fn bench_remove_max<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    if !M::IMMUTABLE {
        return; // Skip for non-immutable maps
    }
    let keys = K::generate(size);
    let values = V::generate(size);
    let map: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        let mut m = map.clone();
        assert!(!m.is_empty());
        for _ in 0..size {
            m = m.without_max().1;
        }
        assert!(m.is_empty())
    })
}

fn bench_insert_once<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let korder = reorder(&keys);
    let vorder = reorder(&values);
    let map: M = keys.clone().into_iter().zip(values).collect();
    b.iter(|| {
        for (k, v) in korder.iter().zip(vorder.iter()).take(100) {
            test::black_box(map.insert_clone(k.clone(), v.clone()));
        }
    })
}

fn bench_remove_once<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let order = reorder(&keys);
    let map: M = keys.clone().into_iter().zip(values).collect();
    b.iter(|| {
        for k in order.iter().take(100) {
            test::black_box(map.remove_clone(k));
        }
    })
}

fn bench_iter<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let m: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        for p in m.iter() {
            test::black_box(p);
        }
    })
}

fn bench_range_iter<M, K, V>(size: usize, b: &mut Bencher)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let order = reorder(&keys);
    let m: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        for k in order.iter().take(10) {
            for p in m.range(k..).take(100) {
                test::black_box(p);
            }
        }
    })
}

// Macro to generate benchmarks for a specific map type and key/value combination
macro_rules! bench_suite {
    ($module:ident, $map:ty, $key:ty, $value:ty) => {
        mod $module {
            use super::*;

            #[bench]
            fn lookup_100(b: &mut Bencher) {
                bench_lookup::<$map, $key, $value>(100, b)
            }

            #[bench]
            fn lookup_1000(b: &mut Bencher) {
                bench_lookup::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn lookup_10000(b: &mut Bencher) {
                bench_lookup::<$map, $key, $value>(10000, b)
            }

            #[bench]
            fn lookup_100000(b: &mut Bencher) {
                bench_lookup::<$map, $key, $value>(100000, b)
            }

            #[bench]
            fn insert_100(b: &mut Bencher) {
                bench_insert::<$map, $key, $value>(100, b)
            }

            #[bench]
            fn insert_1000(b: &mut Bencher) {
                bench_insert::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn insert_mut_100(b: &mut Bencher) {
                bench_insert_mut::<$map, $key, $value>(100, b)
            }

            #[bench]
            fn insert_mut_1000(b: &mut Bencher) {
                bench_insert_mut::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn insert_mut_10000(b: &mut Bencher) {
                bench_insert_mut::<$map, $key, $value>(10000, b)
            }

            #[bench]
            fn insert_mut_100000(b: &mut Bencher) {
                bench_insert_mut::<$map, $key, $value>(100000, b)
            }

            #[bench]
            fn remove_100(b: &mut Bencher) {
                bench_remove::<$map, $key, $value>(100, b)
            }

            #[bench]
            fn remove_1000(b: &mut Bencher) {
                bench_remove::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn remove_10000(b: &mut Bencher) {
                bench_remove::<$map, $key, $value>(10000, b)
            }

            #[bench]
            fn remove_mut_100(b: &mut Bencher) {
                bench_remove_mut::<$map, $key, $value>(100, b)
            }

            #[bench]
            fn remove_mut_1000(b: &mut Bencher) {
                bench_remove_mut::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn remove_mut_10000(b: &mut Bencher) {
                bench_remove_mut::<$map, $key, $value>(10000, b)
            }

            #[bench]
            fn remove_min_1000(b: &mut Bencher) {
                bench_remove_min::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn remove_max_1000(b: &mut Bencher) {
                bench_remove_max::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn insert_once_100(b: &mut Bencher) {
                bench_insert_once::<$map, $key, $value>(100, b)
            }

            #[bench]
            fn insert_once_1000(b: &mut Bencher) {
                bench_insert_once::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn insert_once_10000(b: &mut Bencher) {
                bench_insert_once::<$map, $key, $value>(10000, b)
            }

            #[bench]
            fn remove_once_100(b: &mut Bencher) {
                bench_remove_once::<$map, $key, $value>(100, b)
            }

            #[bench]
            fn remove_once_1000(b: &mut Bencher) {
                bench_remove_once::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn remove_once_10000(b: &mut Bencher) {
                bench_remove_once::<$map, $key, $value>(10000, b)
            }

            #[bench]
            fn remove_once_100000(b: &mut Bencher) {
                bench_remove_once::<$map, $key, $value>(100000, b)
            }

            #[bench]
            fn iter_1000(b: &mut Bencher) {
                bench_iter::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn iter_10000(b: &mut Bencher) {
                bench_iter::<$map, $key, $value>(10000, b)
            }

            #[bench]
            fn range_iter_100(b: &mut Bencher) {
                bench_range_iter::<$map, $key, $value>(100, b)
            }

            #[bench]
            fn range_iter_1000(b: &mut Bencher) {
                bench_range_iter::<$map, $key, $value>(1000, b)
            }

            #[bench]
            fn range_iter_10000(b: &mut Bencher) {
                bench_range_iter::<$map, $key, $value>(10000, b)
            }

            #[bench]
            fn range_iter_100000(b: &mut Bencher) {
                bench_range_iter::<$map, $key, $value>(100000, b)
            }
        }
    };
}

// Generate benchmarks for OrdMap with i64 keys/values
bench_suite!(ordmap_i64, OrdMap<i64, i64>, i64, i64);

// Generate benchmarks for OrdMap with Arc<String> keys/values
bench_suite!(
    ordmap_str,
    OrdMap<Arc<String>, Arc<String>>,
    Arc<String>,
    Arc<String>
);

// // Generate benchmarks for RedBlackTreeMapSync with i64 keys/values
// bench_suite!(rpds_i64, RedBlackTreeMapSync<i64, i64>, i64, i64);

// // Generate benchmarks for RedBlackTreeMapSync with Arc<String> keys/values
// bench_suite!(
//     rpds_str,
//     RedBlackTreeMapSync<Arc<String>, Arc<String>>,
//     Arc<String>,
//     Arc<String>
// );

// // Generate benchmarks for BTreeMap with i64 keys/values
// bench_suite!(btreemap_i64, BTreeMap<i64, i64>, i64, i64);

// // Generate benchmarks for BTreeMap with Arc<String> keys/values
// bench_suite!(
//     btreemap_str,
//     BTreeMap<Arc<String>, Arc<String>>,
//     Arc<String>,
//     Arc<String>
// );
