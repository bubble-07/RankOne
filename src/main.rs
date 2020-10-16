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

use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::view::ContinuousView;
use plotlib::style::{PointMarker, PointStyle, BoxStyle};
use plotlib::repr::{Histogram, HistogramBins};

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

pub fn outer(a : &Array1<f64>, b : &Array1<f64>) -> Array2<f64> {
    einsum("a,b->ab", &[a, b]).unwrap()
          .into_dimensionality::<Ix2>().unwrap()
}

pub fn norm(vec : &Array1<f64>) -> f64 {
    vec.dot(vec).sqrt()
}

pub struct RankOne {
    pub u : Array1<f64>,
    pub v : Array1<f64>
}

impl RankOne {
    pub fn as_mat(&self) -> Array2<f64> {
        outer(&self.u, &self.v)
    }
    pub fn relative_error_to(&self, expected : &RankOne) -> f64 {
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

    let u = &scaled_u_one + &scaled_u_two;
    let v = &scaled_v_one + &scaled_v_two;

    let approx_frob_norm_sq = u.dot(&u) * v.dot(&v);

    let u_one_dot = one.u.dot(&u);
    let u_two_dot = two.u.dot(&u);
    let v_one_dot = one.v.dot(&v);
    let v_two_dot = two.v.dot(&v);

    let full_sum_approx_inner = u_one_dot * v_one_dot + 
                                u_two_dot * v_two_dot;

    let alpha = full_sum_approx_inner / approx_frob_norm_sq;

    let scaled_u = alpha * &u;

    let result = RankOne {
        u : scaled_u,
        v : v
    };
    result
}

/*
fn approx_rank_one_sum(one : &RankOne, two : &RankOne) -> RankOne {
    let u_one_norm = norm(&one.u);
    let v_one_norm = norm(&one.v);
    let u_two_norm = norm(&two.u);
    let v_two_norm = norm(&two.v);

    let u_dot = one.u.dot(&two.u);
    let v_dot = one.v.dot(&two.v);

    let u_sum = &one.u + &two.u;
    let v_sum = &one.v + &two.v;
    let u_sum_norm = norm(&u_sum);
    let v_sum_norm = norm(&v_sum);

    let mult = (u_sum_norm / (u_one_norm + u_two_norm)) * (v_sum_norm / (v_one_norm + v_two_norm));

    let scaled_u_one = v_one_norm * &one.u;
    let scaled_v_one = u_one_norm * &one.v;

    let scaled_u_two = v_two_norm * &two.u;
    let scaled_v_two = u_two_norm * &two.v;

    let total_u = &scaled_u_one + &scaled_u_two;
    let total_v = &scaled_v_one + &scaled_v_two;

    let denominator_sq = u_one_norm * u_one_norm * v_one_norm * v_one_norm +
                         2.0f64 * u_dot * v_dot + 
                         u_two_norm * u_two_norm * v_two_norm * v_two_norm;

    let scale_fac = mult / denominator_sq.sqrt();

    let scaled_total_u = scale_fac * &total_u;
    let result = RankOne {
        u : scaled_total_u,
        v : total_v
    };
    result
}*/

fn full_sum(one : &RankOne, two : &RankOne) -> Array2<f64> {
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
    let num_iters : usize = 10000;
    let mut rng = rand::thread_rng();

    let mut X = Array::zeros((num_iters, 1));
    let mut y = Array::zeros((num_iters, 1));

    let mut scatter_points : Vec<(f64, f64)> = Vec::new();
    let mut rel_errors : Vec<f64> = Vec::new();

    for i in 0..num_iters {
        let mut u_one = Array::zeros((2,));
        u_one[[0,]] = 1.0f64;
        let mut v_one = Array::zeros((2,));
        v_one[[0,]] = 1.0f64;

        let r : f64 = rng.gen();
        let mut t : f64 = rng.gen();
        let mut s : f64 = rng.gen();
        t *= 3.1415f64 / 2.0f64;
        s *= 3.1415f64 / 2.0f64;

        u_one[[0,]] *= r;

        let t_sin = t.sin();
        let s_sin = s.sin();

        let mut u_two = Array::zeros((2,));
        u_two[[0,]] = t.cos();
        u_two[[1,]] = t_sin;

        let mut v_two = Array::zeros((2,));
        v_two[[0,]] = s.cos();
        v_two[[1,]] = s_sin;
       
        let one = RankOne {
            u : u_one,
            v : v_one
        };
        let two = RankOne {
            u : u_two.clone(),
            v : v_two.clone()
        };
        let one_mat = one.as_mat();
        let two_mat = two.as_mat();

        let approx = approx_rank_one_sum(&one, &two);

        let approx_mat = approx.as_mat();

        let actual = best_rank_one_sum(&one, &two);

        let actual_mat = actual.as_mat();

        let rel_error = approx.relative_error_to(&actual);

        //Fill in regression data matrix
        //let estimated_rel_error = 0.25 * (r - r * r) + 0.355 * (1.0 - t.cos()) * (1.0 - s.cos());
        let estimated_rel_error = (1.0 - r) * r * (1.0 - t.cos()) * (1.0 - s.cos());

        //X[[i, 0]] = estimated_rel_error * estimated_rel_error;
        X[[i, 0]] = estimated_rel_error;
        y[[i, 0]] = rel_error;

        scatter_points.push((X[[i, 0]], y[[i, 0]]));
        rel_errors.push(rel_error);

        println!("Relative error: {}", rel_error);
        println!("approx: {}", &approx_mat);
        println!("actual: {}", &actual_mat);
        println!("u two: {}", &u_two);
        println!("v two: {}", &v_two);
        println!("t: {}", t);
        println!("s: {}", s);
        println!("r: {}", r);
        println!("estimated error: {}", estimated_rel_error);
        println!(" ");
    }
    rel_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let XTX = X.t().dot(&X);
    let XTX_inv = XTX.inv().unwrap();
    let XTX_inv_XT = XTX_inv.dot(&X.t()); 
    let B = XTX_inv_XT.dot(&y);
    println!("Regression coefs: {}", B);

    let scatter : Plot = Plot::new(scatter_points).point_style(
        PointStyle::new().size(1.0));

    let view = ContinuousView::new()
        .add(scatter)
        .x_range(0.0, 0.25)
        .y_range(0.0, 0.7)
        .x_label("r(1-r)(1-cos(t))(1-cos(s))")
        .y_label("Relative Error");

    Page::single(&view).save("plots/relative_error.svg").unwrap();

    
    let hist = Histogram::from_slice(&rel_errors, HistogramBins::Count(60))
                    .style(&BoxStyle::new());

    let hist_view = ContinuousView::new()
        .add(hist);

    Page::single(&hist_view).save("plots/relative_error_hist.svg").unwrap();

    let median_ind = rel_errors.len() / 2;
    let median = rel_errors[median_ind];
    println!("Median relative error: {}", median);
}
