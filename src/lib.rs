//! A module for computing homology groups on simplicial complexes

use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView2};

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::iter::zip;

#[derive(PartialEq, Eq, Debug)]
struct Simplex {
    pub vertices: HashSet<usize>,
}

impl Simplex {
    pub fn new(vertices: HashSet<usize>) -> Simplex {
        Simplex { vertices }
    }

    pub fn from_iter<I: Iterator<Item = usize>>(iter: I) -> Simplex {
        Simplex::new(HashSet::from_iter(iter))
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
                collected_boundaries.insert(Simplex::from_iter(
                    boundary.iter().map(|index| row[index.clone()]),
                ));
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
    pub fn new(complex: SimplicialComplex) -> Result<ChainComplex, String> {
        let mut boundary_operators = vec![];

        if complex.simplices.len() > 1 {
            for i_complex in 0..complex.simplices.len() - 1 {
                boundary_operators.push(
                    match ChainComplex::boundary_operator(
                        complex.simplices[i_complex].view(),
                        complex.simplices[i_complex + 1].view(),
                    ) {
                        Ok(op) => op,
                        Err(message) => return Err(message),
                    },
                );
            }
        }

        Ok(ChainComplex {
            chain_groups: complex.simplices,
            boundary_operators: boundary_operators,
        })
    }

    /// construct the boundary operator from one d-dimensional chain group to the d-1 dimensional
    /// chain group
    fn boundary_operator(
        chain_group_top: ArrayView2<usize>,
        chain_group_bottom: ArrayView2<usize>,
    ) -> Result<Array2<i32>, String> {
        if chain_group_top.shape()[0] == 0 || chain_group_bottom.shape()[0] == 0 {
            return Ok(Array2::zeros((0, 0)));
        }

        let top_dimension: i32 = chain_group_top.shape()[1] as i32 - 1;
        let bottom_dimension: i32 = chain_group_bottom.shape()[1] as i32 - 1;
        if top_dimension < 0 || bottom_dimension < 0 || bottom_dimension + 1 != top_dimension {
            return Err("Problem in dimensions of provided chain groups. Cannot compute a boundary operator.".to_string());
        }

        let top_dimension = top_dimension as usize;
        let bottom_dimension = bottom_dimension as usize;

        let mut index = HashMap::new();
        for (i_bottom, row) in chain_group_bottom.rows().into_iter().enumerate() {
            index.insert(
                Simplex::from_iter(row.into_iter().map(|element| *element)),
                i_bottom,
            );
        }

        let reference_stencil: Vec<(Vec<usize>, i32)> = (0..top_dimension + 1)
            .combinations(bottom_dimension + 1)
            .collect_vec()
            .into_iter()
            .rev()
            .enumerate()
            .map(|(index, pattern)| (pattern, (-1 as i32).pow(index as u32)))
            .collect_vec();

        let mut operator =
            Array2::zeros((chain_group_bottom.shape()[0], chain_group_top.shape()[0]));

        for (i_col, element) in chain_group_top.rows().into_iter().enumerate() {
            for (stencil, value) in &reference_stencil {
                let boundary = stencil
                    .iter()
                    .map(|i_element| element[*i_element])
                    .collect_vec();

                let i_row = match index.get(&Simplex::from_iter(boundary.clone().into_iter())) {
                    Some(i_row) => *i_row,
                    None => {
                        return Err(format!(
                            "Found boundary {:?} of simplex {:?} that was not in provided bottom chain.",
                            boundary, i_col
                        ));
                    }
                };

                let bottom_element = chain_group_bottom
                    .row(i_row)
                    .into_iter()
                    .map(|val| *val)
                    .collect_vec();

                // Determine orientation of boundary with respect to bottom_element
                let mut mismatches: HashMap<usize, usize> = HashMap::new();
                for (bound, bottom) in zip(boundary, bottom_element) {
                    if bound != bottom {
                        mismatches.insert(bottom, bound);
                    }
                }

                let mut swaps = Vec::new();
                while !mismatches.is_empty() {
                    let (key, val) = match mismatches.iter().next()
                    {
                        Some((key, val)) => (key.clone(), val.clone()),
                        None => return Err("Got to end of mismtatches while checking orientation without mismatches being empty.".to_string()),
                    };

                    let reciprocal = match mismatches.get(&val)
                    {
                        Some(reciprocal) => reciprocal.clone(),
                        None => return Err("Found value in mismatches that has no reciprocal while checking orientations.".to_string())
                    };

                    swaps.push((val, reciprocal));
                    mismatches.remove(&key);
                    mismatches.remove(&val);
                    if key != reciprocal {
                        mismatches.insert(key, reciprocal);
                    }
                }

                operator[[i_row, i_col]] += (-1 as i32).pow(swaps.len() as u32) * value;
            }
        }

        Ok(operator)
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
                .map(|row| Simplex::from_iter(row.into_iter().map(|val| *val))),
        )
    }

    #[test]
    fn test_simplicial_complex_from_curve() {
        let curve = array![[0, 1], [1, 2], [2, 3], [3, 4]];
        let complex = SimplicialComplex::from(curve.clone());
        assert!(complex.simplices.len() == 2);

        assert_eq!(to_unordered(&complex.simplices[0]), to_unordered(&curve));

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

        assert_eq!(to_unordered(&complex.simplices[0]), to_unordered(&curve));

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

        assert_eq!(to_unordered(&complex.simplices[0]), to_unordered(&tetra));

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

    #[test]
    fn test_chain_complex_from_curve() {
        let curve = array![[0, 1], [1, 2], [2, 3], [3, 4]];

        let complex_outcome = ChainComplex::new(SimplicialComplex::from(curve.clone()));
        assert!(complex_outcome.is_ok());

        let complex = complex_outcome.unwrap();
        assert!(complex.chain_groups.len() == 2);
        assert!(complex.boundary_operators.len() == 1);

        let mut reference = Array1::zeros(5);
        reference[[complex.chain_groups[1]
            .rows()
            .into_iter()
            .map(|row| Simplex::from_iter(row.into_iter().map(|val| *val)))
            .position(|simplex| simplex == Simplex::from_iter(0..1))
            .unwrap()]] += -1;
        reference[[complex.chain_groups[1]
            .rows()
            .into_iter()
            .map(|row| Simplex::from_iter(row.into_iter().map(|val| *val)))
            .position(|simplex| simplex == Simplex::from_iter(4..5))
            .unwrap()]] += 1;
        assert_eq!(
            complex.boundary_operators[0].dot(&array![1, 1, 1, 1]),
            reference
        );
    }

    #[test]
    fn test_chain_complex_from_tetrahedra() {
        let tetra = array![[0, 1, 2, 3]];

        let complex_outcome = ChainComplex::new(SimplicialComplex::from(tetra.clone()));
        assert!(complex_outcome.is_ok());

        let complex = complex_outcome.unwrap();
        assert!(complex.chain_groups.len() == 4);
        assert!(complex.boundary_operators.len() == 3);

        assert_eq!(
            Array2::<i32>::zeros((
                complex.boundary_operators[1].shape()[0],
                complex.boundary_operators[0].shape()[1]
            )),
            &complex.boundary_operators[1].dot(&complex.boundary_operators[0])
        );

        assert_eq!(
            Array2::<i32>::zeros((
                complex.boundary_operators[2].shape()[0],
                complex.boundary_operators[1].shape()[1]
            )),
            &complex.boundary_operators[2].dot(&complex.boundary_operators[1])
        );
    }
}
