use ndarray::{Array2, ArrayView2};

struct SimplicialComplex {
    pub simplices: Vec<Array2<usize>>,
}

impl SimplicialComplex {
    pub fn from(top_level_simplices: Array2<usize>) -> SimplicialComplex {
        SimplicialComplex {
            simplices: vec![top_level_simplices],
        }
    }
}

struct ChainComplex {
    pub chain_groups: Vec<Array2<usize>>,
    pub boundary_operators: Vec<Array2<i32>>,
}

impl ChainComplex {
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

    pub fn boundary_operator(
        chain_group_top: ArrayView2<usize>,
        chain_group_bottom: ArrayView2<usize>,
    ) -> Array2<i32> {
        let mut operator =
            Array2::zeros((chain_group_bottom.shape()[0], chain_group_top.shape()[0]));
        operator
    }
}

#[cfg(test)]
mod tests {}
