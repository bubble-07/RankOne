#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_parens)]
#![allow(unused_variables)]


extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use rand::prelude::*;
use ndarray_einsum_beta::*;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

pub fn outer(a : &Array1<f32>, b : &Array1<f32>) -> Array2<f32> {
    einsum("a,b->ab", &[a, b]).unwrap()
          .into_dimensionality::<Ix2>().unwrap()
}

pub fn norm(vec : &Array1<f32>) -> f32 {
    vec.dot(vec).sqrt()
}

pub struct RankOne {
    pub u : Array1<f32>,
    pub v : Array1<f32>
}

impl RankOne {
    pub fn as_mat(&self) -> Array2<f32> {
        outer(&self.u, &self.v)
    }
    pub fn relative_error_to(&self, expected : &RankOne) -> f32 {
        let actual_mat = self.as_mat();
        let expected_mat = expected.as_mat();

        let diff = &expected_mat - &actual_mat;
        let diff_norm = diff.opnorm_fro().unwrap();
        
        let expected_norm = expected_mat.opnorm_fro().unwrap();

        let rel_error = diff_norm / expected_norm;

        rel_error
    }
}

fn approx_rank_one_sum(one : &RankOne, two : &RankOne) -> RankOne {
    let u_one_norm = norm(&one.u);
    let v_one_norm = norm(&one.v);
    let u_two_norm = norm(&two.u);
    let v_two_norm = norm(&two.v);

    let u_dot = one.u.dot(&two.u);
    let v_dot = one.v.dot(&two.v);

    let scaled_u_one = v_one_norm * &one.u;
    let scaled_v_one = u_one_norm * &one.v;

    let scaled_u_two = v_two_norm * &two.u;
    let scaled_v_two = u_two_norm * &two.v;

    let total_u = &scaled_u_one + &scaled_u_two;
    let total_v = &scaled_v_one + &scaled_v_two;

    let denominator_sq = u_one_norm * u_one_norm * v_one_norm * v_one_norm +
                         2.0f32 * u_dot * v_dot + 
                         u_two_norm * u_two_norm * v_two_norm * v_two_norm;

    let scale_fac = 1.0f32 / denominator_sq.sqrt();

    let scaled_total_u = scale_fac * &total_u;
    let result = RankOne {
        u : scaled_total_u,
        v : total_v
    };
    result
}

fn full_sum(one : &RankOne, two : &RankOne) -> Array2<f32> {
    &one.as_mat() + &two.as_mat()    
}

fn best_rank_one_sum(one : &RankOne, two : &RankOne) -> RankOne {
    let mat = full_sum(one, two);
    let (maybe_u, sigma, maybe_v_t) = mat.svd(true, true).unwrap();
    if let Option::Some(u) = maybe_u {
        if let Option::Some(v_t) = maybe_v_t {
            let u_vec = u.column(0);
            let v_vec = v_t.row(0);
            let s = sigma[[0,]];
            let u_scaled = s * &u_vec;

            let result = RankOne {
                u : u_scaled.clone(),
                v : v_vec.clone().to_owned()
            };
            result
        } else {
            panic!();
        }
    } else {
        println!("Rank one decomposition failed for {}", mat);
        panic!();
    }
}

fn main() {
    let num_iters : usize = 1000;
    let dim : usize = 2;
    let mut worst_rel_error = 0.0f32;

    for i in 0..num_iters {
        let mut u_one = Array::random((dim,), StandardNormal);
        let mut v_one = Array::random((dim,), StandardNormal);
        let u_two = Array::random((dim,), StandardNormal);
        let v_two = Array::random((dim,), StandardNormal);

        let u_dot = u_one.dot(&u_two);

        if (u_dot < 0.0f32) {
            u_one *= -1.0f32;
        }

        let v_dot = v_one.dot(&v_two);

        if (v_dot < 0.0f32) {
            v_one *= -1.0f32;
        }

        let one = RankOne {
            u : u_one,
            v : v_one
        };
        let two = RankOne {
            u : u_two,
            v : v_two
        };
        let one_mat = one.as_mat();
        let two_mat = two.as_mat();

        let approx = approx_rank_one_sum(&one, &two);

        let approx_mat = approx.as_mat();

        let actual = best_rank_one_sum(&one, &two);

        let actual_mat = actual.as_mat();

        let rel_error = approx.relative_error_to(&actual);

        if (rel_error > worst_rel_error) {
            worst_rel_error = rel_error;
            println!("Relative error: {}", rel_error);
            println!("Mat one: {}", &one_mat);
            println!("Mat two: {}", &two_mat);
            println!("approx: {}", &approx_mat);
            println!("actual: {}", &actual_mat);
            println!("u dot: {}", u_dot);
            println!("v dot: {}", v_dot);
            println!("combined dot: {}", u_dot * v_dot);
            println!(" ");
        }
    }
}
