// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![feature(test)]

extern crate imbl;
extern crate rand;
extern crate rpds;
extern crate test;

mod common;

use std::borrow::Borrow;
use std::collections::HashMap as StdHashMap;
use std::hash::Hash;
use std::iter::FromIterator;
use std::sync::Arc;
use test::Bencher;

use archery::ArcTK;
use common::{reorder, TestData};
use imbl::hashmap::HashMap;
use rpds::HashTrieMapSync;

// Trait to abstract over different map implementations
trait BenchMap<K, V>: Clone + FromIterator<(K, V)>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    const IMMUTABLE: bool = true;
    type Iter<'a>: Iterator<Item = (&'a K, &'a V)>
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
        Q: Hash + Eq + ?Sized;
    fn iter(&self) -> Self::Iter<'_>;
    #[allow(dead_code)]
    fn is_empty(&self) -> bool;
}

// Implementation for imbl::HashMap
impl<K, V> BenchMap<K, V> for HashMap<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    type Iter<'a>
        = imbl::hashmap::Iter<'a, K, V, imbl::shared_ptr::DefaultSharedPtr>
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        HashMap::new()
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
        Q: Hash + Eq + ?Sized,
    {
        self.get(k)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

// Implementation for std::collections::HashMap
impl<K, V> BenchMap<K, V> for StdHashMap<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    const IMMUTABLE: bool = false;
    type Iter<'a>
        = std::collections::hash_map::Iter<'a, K, V>
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        StdHashMap::new()
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
        Q: Hash + Eq + ?Sized,
    {
        self.get(k)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

// Implementation for rpds::HashTrieMapSync
impl<K, V> BenchMap<K, V> for HashTrieMapSync<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    type Iter<'a>
        = rpds::map::hash_trie_map::Iter<'a, K, V, ArcTK>
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        HashTrieMapSync::new_sync()
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
        Q: Hash + Eq + ?Sized,
    {
        self.get(k)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn is_empty(&self) -> bool {
        self.size() == 0
    }
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
            test::black_box(m.get(k));
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
        m
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
        m
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
        m
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
        }
    };
}

// Generate benchmarks for imbl::HashMap with i64 keys/values
bench_suite!(hashmap_i64, HashMap<i64, i64>, i64, i64);

// Generate benchmarks for imbl::HashMap with Arc<String> keys/values
bench_suite!(
    hashmap_str,
    HashMap<Arc<String>, Arc<String>>,
    Arc<String>,
    Arc<String>
);

// // Generate benchmarks for std::collections::HashMap with i64 keys/values
// bench_suite!(stdhashmap_i64, StdHashMap<i64, i64>, i64, i64);

// // Generate benchmarks for std::collections::HashMap with Arc<String> keys/values
// bench_suite!(
//     stdhashmap_str,
//     StdHashMap<Arc<String>, Arc<String>>,
//     Arc<String>,
//     Arc<String>
// );

// Generate benchmarks for rpds::HashTrieMapSync with i64 keys/values
// bench_suite!(rpds_i64, HashTrieMapSync<i64, i64>, i64, i64);

// // Generate benchmarks for rpds::HashTrieMapSync with Arc<String> keys/values
// bench_suite!(
//     rpds_str,
//     HashTrieMapSync<Arc<String>, Arc<String>>,
//     Arc<String>,
//     Arc<String>
// );
