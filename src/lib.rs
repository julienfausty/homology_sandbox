//! A module for computing homology groups on simplicial complexes

use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView2, s};

use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::iter::zip;

/// Utility structure mostly for evaluating equality of simplices
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

/// return the reduced row echelon form of the provided matrix along with the change of basis
/// matrix that takes the original matrix to the reduced form
fn reduced_row_echelon_form(matrix: &Array2<i32>) -> (Array2<f64>, Array2<f64>) {
    let mut buffer_matrix = Array2::from_shape_vec(
        [matrix.shape()[0], matrix.shape()[1]],
        matrix.iter().map(|val| *val as f64).collect::<Vec<f64>>(),
    )
    .unwrap();
    let mut change_of_basis = Array2::eye(matrix.shape()[0]);

    let number_of_rows = buffer_matrix.shape()[0];
    let number_of_columns = buffer_matrix.shape()[1];
    let max_number_of_pivots = min(number_of_rows, number_of_columns);
    let max_number_of_iterations = max(number_of_rows, number_of_columns);

    if max_number_of_pivots == 0 {
        return (buffer_matrix, change_of_basis);
    }

    let mut row_offset = 0;

    for i_row in 0..max_number_of_iterations {
        let i_pivot = i_row - row_offset;

        if i_pivot == number_of_rows || i_row == number_of_columns {
            break;
        }

        let pivot_diagonal = buffer_matrix[[i_pivot, i_row]];

        if pivot_diagonal == 0.0 {
            // look for next row with non-zero value and swap
            let mut found = false;
            for j_row in (i_pivot + 1)..number_of_rows {
                found = buffer_matrix[[j_row, i_row]] != 0.0;
                if found {
                    let pivot_row_buffer = buffer_matrix.row(i_pivot).to_owned();
                    let j_row_buffer = buffer_matrix.row(j_row).to_owned();

                    buffer_matrix.row_mut(i_pivot).assign(&j_row_buffer);
                    buffer_matrix.row_mut(j_row).assign(&pivot_row_buffer);

                    let pivot_row_buffer = change_of_basis.row(i_pivot).to_owned();
                    let j_row_buffer = change_of_basis.row(j_row).to_owned();

                    change_of_basis.row_mut(i_pivot).assign(&j_row_buffer);
                    change_of_basis.row_mut(j_row).assign(&pivot_row_buffer);

                    break;
                }
            }

            if !found {
                row_offset += 1;
                continue;
            }
        }

        let pivot_diagonal = buffer_matrix[[i_pivot, i_row]];

        for value in buffer_matrix.row_mut(i_pivot).iter_mut() {
            *value /= pivot_diagonal;
        }

        for value in change_of_basis.row_mut(i_pivot).iter_mut() {
            *value /= pivot_diagonal;
        }

        let pivot_vector = buffer_matrix.row(i_pivot).to_owned();
        let basis_pivot_vector = change_of_basis.row(i_pivot).to_owned();

        for j_row in 0..number_of_rows {
            if j_row == i_pivot {
                continue;
            }

            let multiplier = buffer_matrix[[j_row, i_row]];
            for (value, pivot) in zip(buffer_matrix.row_mut(j_row).iter_mut(), pivot_vector.iter())
            {
                *value -= pivot * multiplier;
            }

            for (value, pivot) in zip(
                change_of_basis.row_mut(j_row).iter_mut(),
                basis_pivot_vector.iter(),
            ) {
                *value -= pivot * multiplier;
            }
        }
    }

    let first_non_zero = match buffer_matrix
        .row(max_number_of_pivots - 1)
        .iter()
        .find(|&&val| val != 0.0)
    {
        Some(reference) => Some(reference.clone()),
        None => None,
    };

    match first_non_zero {
        Some(first_non_zero) => {
            if first_non_zero != 1.0 {
                for value in buffer_matrix.row_mut(max_number_of_pivots - 1).iter_mut() {
                    *value = *value / first_non_zero;
                }

                for value in change_of_basis.row_mut(max_number_of_pivots - 1).iter_mut() {
                    *value = *value / first_non_zero;
                }
            }
        }
        None => (),
    };

    (buffer_matrix, change_of_basis)
}

#[derive(Debug)]
pub struct HomologyGroup {
    pub cycle_basis: Array2<i32>,
    pub boundary_basis: Array2<i32>,
}

impl HomologyGroup {
    pub fn new(top_boundary_map: &Array2<i32>, bottom_boundary_map: &Array2<i32>) -> HomologyGroup {
        let (top_rref, top_basis_change) = reduced_row_echelon_form(top_boundary_map);

        let first_zero_row = match top_rref
            .rows()
            .into_iter()
            .map(|row| row.into_iter().map(|value| value.powf(2.0)).sum())
            .position(|norm: f64| norm < 1e-12)
        {
            Some(i_row) => i_row,
            None => top_rref.shape()[0],
        };

        let boundary_basis = Array2::from_shape_vec(
            (first_zero_row, top_basis_change.shape()[1]),
            top_basis_change
                .slice(s![..first_zero_row, ..])
                .into_iter()
                .map(|value| *value as i32)
                .collect(),
        )
        .unwrap();

        let (bottom_rref, bottom_basis_change) =
            reduced_row_echelon_form(&bottom_boundary_map.t().to_owned());
        let bottom_rref = bottom_rref.t().to_owned();
        let bottom_basis_change = bottom_basis_change.t().to_owned();

        let first_zero_col = match bottom_rref
            .columns()
            .into_iter()
            .map(|col| col.into_iter().map(|value| value.powf(2.0)).sum())
            .position(|norm: f64| norm < 1e-12)
        {
            Some(i_col) => i_col,
            None => bottom_rref.shape()[1],
        };

        let cycle_basis = Array2::from_shape_vec(
            (
                bottom_basis_change.shape()[0],
                bottom_rref.shape()[1] - first_zero_col,
            ),
            bottom_basis_change
                .slice(s![.., first_zero_col..])
                .into_iter()
                .map(|value| *value as i32)
                .collect(),
        )
        .unwrap()
        .t()
        .to_owned();

        HomologyGroup {
            cycle_basis,
            boundary_basis,
        }
    }

    pub fn from_chain_complex(complex: &ChainComplex) -> Vec<HomologyGroup> {
        if complex.boundary_operators.is_empty() {
            return Vec::new();
        }

        let number_boundary_operators = complex.boundary_operators.len();

        let mut groups = vec![HomologyGroup::new(
            &Array2::zeros((complex.chain_groups[0].shape()[0], 1)),
            &complex.boundary_operators[0],
        )];

        groups.extend(
            zip(
                complex.boundary_operators[..(number_boundary_operators - 1)].iter(),
                complex.boundary_operators[1..].iter(),
            )
            .map(|(top, bottom)| HomologyGroup::new(top, bottom)),
        );

        groups.push(HomologyGroup::new(
            &complex.boundary_operators[number_boundary_operators - 1],
            &Array2::zeros((1, complex.chain_groups.last().unwrap().shape()[0])),
        ));

        groups.into_iter().rev().collect()
    }

    pub fn betti_number(&self) -> usize {
        self.cycle_basis.shape()[0] - self.boundary_basis.shape()[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use rand::Rng;

    fn to_unordered(ordered: &Array2<usize>) -> HashSet<Simplex> {
        HashSet::from_iter(
            ordered
                .rows()
                .into_iter()
                .map(|row| Simplex::from_iter(row.into_iter().map(|val| *val))),
        )
    }

    fn strip() -> Array2<usize> {
        array![
            [0, 1, 2],
            [2, 1, 3],
            [2, 3, 4],
            [4, 3, 0],
            [0, 3, 5],
            [5, 1, 0]
        ]
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
    fn test_simplicial_complex_from_strip() {
        let strip = strip();
        let complex = SimplicialComplex::from(strip.clone());
        assert!(complex.simplices.len() == 3);

        assert_eq!(to_unordered(&complex.simplices[0]), to_unordered(&strip));

        assert_eq!(
            to_unordered(&complex.simplices[1]),
            to_unordered(&array![
                [0, 1],
                [1, 2],
                [0, 2],
                [2, 3],
                [1, 3],
                [3, 4],
                [2, 4],
                [0, 3],
                [0, 4],
                [0, 5],
                [3, 5],
                [1, 5]
            ])
        );

        assert_eq!(
            to_unordered(&complex.simplices[2]),
            to_unordered(&array![[0], [1], [2], [3], [4], [5]])
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

    #[test]
    fn test_chain_complex_from_strip() {
        let strip = strip();

        let complex_outcome = ChainComplex::new(SimplicialComplex::from(strip.clone()));
        assert!(complex_outcome.is_ok());

        let complex = complex_outcome.unwrap();
        assert!(complex.chain_groups.len() == 3);
        assert!(complex.boundary_operators.len() == 2);

        assert_eq!(
            Array2::<i32>::zeros((
                complex.boundary_operators[1].shape()[0],
                complex.boundary_operators[0].shape()[1]
            )),
            &complex.boundary_operators[1].dot(&complex.boundary_operators[0])
        );
    }

    #[test]
    fn test_reduced_row_echelon_form_random() {
        let mut rng = rand::rng();
        let random_matrix = Array2::random(
            (rng.random_range(1..50), rng.random_range(1..50)),
            Uniform::new(0, 10),
        );
        let (rref, basis) = reduced_row_echelon_form(&random_matrix);
        for i_row in 0..rref.shape()[0] {
            let max_j_col = min(i_row, rref.shape()[1] - 1);
            for j_col in 0..max_j_col {
                assert_eq!(rref[[i_row, j_col]], 0.0);
            }
            let diagonal = rref[[i_row, max_j_col]];
            assert!(diagonal == 1.0 || diagonal == 0.0);
        }

        let float_matrix = Array2::from_shape_vec(
            [random_matrix.shape()[0], random_matrix.shape()[1]],
            random_matrix
                .iter()
                .map(|val| *val as f64)
                .collect::<Vec<f64>>(),
        )
        .unwrap();

        let change = basis.dot(&float_matrix);
        let _ = zip(rref.iter(), change.iter())
            .map(|(lhs, rhs)| assert!((*lhs - *rhs).abs() < 1e-8))
            .collect::<Vec<_>>();
    }

    #[test]
    fn test_homology_group_curve() {
        let curve = array![[0, 1], [1, 2], [2, 3], [3, 4]];

        let complex = ChainComplex::new(SimplicialComplex::from(curve)).unwrap();

        let homology_group = HomologyGroup::new(
            &complex.boundary_operators[0],
            &Array2::zeros((1, complex.chain_groups[1].shape()[0])),
        );

        assert!(homology_group.betti_number() == 1);
    }

    #[test]
    fn test_homology_group_tetra() {
        let tetra = array![[0, 1, 2, 3]];

        let complex = ChainComplex::new(SimplicialComplex::from(tetra)).unwrap();

        let homology_group = HomologyGroup::new(
            &complex.boundary_operators[1],
            &complex.boundary_operators[2],
        );

        assert!(homology_group.betti_number() == 0);
    }

    #[test]
    fn test_homology_group_sphere() {
        let tetra = array![[0, 1, 2, 3]];

        let complex = ChainComplex::new(SimplicialComplex::from(tetra)).unwrap();

        let homology_group = HomologyGroup::new(
            &Array2::zeros((complex.chain_groups[1].shape()[0], 1)),
            &complex.boundary_operators[1],
        );

        assert_eq!(homology_group.betti_number(), 1);
    }

    #[test]
    fn test_homology_groups_circle() {
        let curve = array![[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]];

        let complex = ChainComplex::new(SimplicialComplex::from(curve)).unwrap();

        let homology_groups = HomologyGroup::from_chain_complex(&complex);

        assert_eq!(homology_groups.len(), 2);

        assert_eq!(homology_groups[0].betti_number(), 1);
        assert_eq!(homology_groups[1].betti_number(), 1);
    }

    #[test]
    fn test_homology_groups_tetra() {
        let tetra = array![[0, 1, 2, 3]];

        let complex = ChainComplex::new(SimplicialComplex::from(tetra)).unwrap();

        let homology_groups = HomologyGroup::from_chain_complex(&complex);

        assert_eq!(homology_groups.len(), 4);
        assert_eq!(homology_groups[0].betti_number(), 1);
        assert_eq!(homology_groups[1].betti_number(), 0);
        assert_eq!(homology_groups[2].betti_number(), 0);
        assert_eq!(homology_groups[3].betti_number(), 0);
    }

    #[test]
    fn test_homology_groups_strip() {
        let strip = strip();

        let complex = ChainComplex::new(SimplicialComplex::from(strip)).unwrap();

        let homology_groups = HomologyGroup::from_chain_complex(&complex);

        assert_eq!(homology_groups.len(), 3);
        assert_eq!(homology_groups[0].betti_number(), 1);
        assert_eq!(homology_groups[1].betti_number(), 1);
        assert_eq!(homology_groups[2].betti_number(), 0);
    }

    #[test]
    fn test_homology_groups_torus() {
        let torus = array![
            [0, 7, 3],
            [3, 8, 4],
            [4, 2, 0],
            [2, 0, 7],
            [7, 3, 8],
            [8, 4, 2],
            [2, 5, 7],
            [7, 6, 8],
            [8, 1, 2],
            [1, 2, 5],
            [5, 7, 6],
            [6, 8, 1],
            [1, 3, 5],
            [5, 4, 6],
            [6, 0, 1],
            [1, 0, 3],
            [5, 3, 4],
            [6, 4, 0]
        ];

        let complex = ChainComplex::new(SimplicialComplex::from(torus)).unwrap();

        let homology_groups = HomologyGroup::from_chain_complex(&complex);

        assert_eq!(homology_groups.len(), 3);
        assert_eq!(homology_groups[0].betti_number(), 1);
        assert_eq!(homology_groups[1].betti_number(), 2);
        assert_eq!(homology_groups[2].betti_number(), 1);
    }
}
