//! A module for computing homology groups on simplicial complexes

use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

#[derive(PartialEq, Eq, Debug)]
struct Simplex {
    pub vertices: HashSet<usize>,
}

impl Simplex {
    pub fn new(vertices: HashSet<usize>) -> Simplex {
        Simplex { vertices }
    }
}

impl Hash for Simplex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let _ = self.vertices.iter().map(|element| element.hash(state));
    }
}

/// A structure representing a simplicial complex
pub struct SimplicialComplex {
    /// Each element of the vector contains the simplex topology information at a given dimension
    /// starting with the highest. As such, each element is a 2 dimensional array with shape
    /// (number_of_simplices, dimension + 1).
    pub simplices: Vec<Array2<usize>>,
}

impl SimplicialComplex {
    /// A utility method for constructing the entire simplicial complex from the topology
    /// information at the highest dimension
    pub fn from(top_level_simplices: Array2<usize>) -> SimplicialComplex {
        if top_level_simplices.shape()[0] == 0 || top_level_simplices.shape()[1] == 0 {
            return SimplicialComplex { simplices: vec![] };
        }

        let top_dim = top_level_simplices.shape()[1] - 1;
        let mut simplices = vec![top_level_simplices];
        for i_dim in 0..top_dim {
            simplices.push(SimplicialComplex::map_boundaries(simplices[i_dim].view()));
        }

        SimplicialComplex {
            simplices: simplices,
        }
    }

    fn map_boundaries(simplices: ArrayView2<usize>) -> Array2<usize> {
        let high_dimension = simplices.shape()[1] - 1;
        if high_dimension <= 0 {
            return Array2::zeros((0, 0));
        }
        let low_dimension = high_dimension - 1;

        let reference_boundaries: Vec<Vec<usize>> = (0..high_dimension + 1)
            .combinations(low_dimension + 1)
            .collect();

        let mut collected_boundaries = HashSet::new();

        for row in simplices.rows() {
            for boundary in reference_boundaries.iter() {
                collected_boundaries.insert(Simplex::new(HashSet::from_iter(
                    boundary.iter().map(|index| row[index.clone()]),
                )));
            }
        }

        let mut boundaries = Array2::zeros((collected_boundaries.len(), low_dimension + 1));
        for (i_simplex, simplex) in collected_boundaries.into_iter().enumerate() {
            boundaries
                .row_mut(i_simplex)
                .assign(&Array1::from_vec(simplex.vertices.into_iter().collect()));
        }

        boundaries
    }
}

/// A structure representing a chain complex
pub struct ChainComplex {
    /// Each element of the vector contains the basis elements of the associated groups organized
    /// in decreasing dimension
    pub chain_groups: Vec<Array2<usize>>,
    /// Each element of the vector contains a linear operator acting on vectors in the associated
    /// group in the chain_groups and taking them to the next chain_group in the sequence.
    ///
    /// d_k : C_k -> C_k-1
    pub boundary_operators: Vec<Array2<i32>>,
}

impl ChainComplex {
    /// Construct a chain complex from a SimplicialComplex
    pub fn new(complex: SimplicialComplex) -> ChainComplex {
        let mut boundary_operators = vec![];

        if complex.simplices.len() > 1 {
            for i_complex in 0..complex.simplices.len() - 1 {
                boundary_operators.push(ChainComplex::boundary_operator(
                    complex.simplices[i_complex + 1].view(),
                    complex.simplices[i_complex].view(),
                ));
            }
        }

        ChainComplex {
            chain_groups: complex.simplices,
            boundary_operators: boundary_operators,
        }
    }

    /// construct the boundary operator from one d-dimensional chain group to the d-1 dimensional
    /// chain group
    fn boundary_operator(
        chain_group_top: ArrayView2<usize>,
        chain_group_bottom: ArrayView2<usize>,
    ) -> Array2<i32> {
        let mut operator =
            Array2::zeros((chain_group_bottom.shape()[0], chain_group_top.shape()[0]));
        operator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    fn to_unordered(ordered: &Array2<usize>) -> HashSet<Simplex> {
        HashSet::from_iter(
            ordered
                .rows()
                .into_iter()
                .map(|row| Simplex::new(HashSet::from_iter(row.into_iter().map(|val| *val)))),
        )
    }

    #[test]
    fn test_simplicial_complex_from_curve() {
        let curve = array![[0, 1], [1, 2], [2, 3], [3, 4]];
        let complex = SimplicialComplex::from(curve.clone());
        assert!(complex.simplices.len() == 2);
        // TODO: make these unordered equals
        assert_eq!(to_unordered(&complex.simplices[0]), to_unordered(&curve));
        // TODO: make these unordered equals
        assert_eq!(
            to_unordered(&complex.simplices[1]),
            to_unordered(&array![[0], [1], [2], [3], [4]])
        );
    }

    #[test]
    fn test_simplicial_complex_from_circle() {
        let curve = array![[0, 1], [1, 2], [2, 3], [3, 0]];
        let complex = SimplicialComplex::from(curve.clone());
        assert!(complex.simplices.len() == 2);
        // TODO: make these unordered equals
        assert_eq!(to_unordered(&complex.simplices[0]), to_unordered(&curve));
        // TODO: make these unordered equals
        assert_eq!(
            to_unordered(&complex.simplices[1]),
            to_unordered(&array![[0], [1], [2], [3]])
        );
    }

    #[test]
    fn test_simplicial_complex_from_tetrahedra() {
        let tetra = array![[0, 1, 2, 3]];
        let complex = SimplicialComplex::from(tetra.clone());
        assert!(complex.simplices.len() == 4);
        // TODO: make these unordered equals
        assert_eq!(to_unordered(&complex.simplices[0]), to_unordered(&tetra));
        // TODO: make these unordered equals
        assert_eq!(
            to_unordered(&complex.simplices[1]),
            to_unordered(&array![[0, 1, 2], [1, 2, 3], [0, 1, 3], [2, 3, 0]])
        );
        assert_eq!(
            to_unordered(&complex.simplices[2]),
            to_unordered(&array![[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]])
        );
        assert_eq!(
            to_unordered(&complex.simplices[3]),
            to_unordered(&array![[0], [1], [2], [3]])
        );
    }
}
