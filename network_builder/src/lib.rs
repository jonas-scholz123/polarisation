use pyo3::prelude::*;
use ndarray::Array2;
use std::collections::HashMap;

fn compute_overlaps_arr(author_vec: Vec<HashMap<usize, i32>>, sub_total_comments:HashMap<usize, i32>) -> Vec<Vec<f32>>{
    // First, we make an index:

    let nr_subs = sub_total_comments.len();
    let mut overlaps = Array2::<f32>::zeros((nr_subs, nr_subs));
    let mut author_counter = 0;

    let total_nr_authors = author_vec.len() as f32;

    for subreddit_comments in author_vec {

        let total_comments_by_author:f32 = subreddit_comments.values().sum::<i32>() as f32;

        for (&sub1, &nr_comments_1) in &subreddit_comments {
            for (&sub2, &nr_comments_2) in &subreddit_comments {
                if sub1 != sub2 {
                    overlaps[[sub1, sub2]] += (nr_comments_1 * nr_comments_2) as f32 / (total_comments_by_author * sub_total_comments[&sub2] as f32);
                }
            }
        }
        author_counter += 1;
        if author_counter % 500000 == 0 {
            println!("nr authors: {} {} % done", author_counter, 100. * (author_counter as f32) / total_nr_authors);
        }
    }

    let mut overlaps_vec = vec![vec![0.; nr_subs]; nr_subs];

    for i in 0..nr_subs {
        for j in 0..nr_subs {
            // Make symmetric.
            overlaps_vec[i][j] = overlaps[[j, i]].min(overlaps[[i, j]]);
            if overlaps_vec[i][j] > 0.5 {
                println!("sub ids: {} {}, overlap: {} nr_subs: {} {}", i, j, overlaps_vec[i][j], sub_total_comments[&i], sub_total_comments[&j])
            }
        }
    }

    println!("finished generating network matrix. Passing back to python.");
    overlaps_vec
}

fn make_author_vec(authors: Vec<&str>, subreddits: Vec<usize>, counts: Vec<i32>) -> (Vec<HashMap<usize, i32>>, HashMap<usize, i32>) {

    // Keep track of the total number of comments in a subreddit.
    let mut sub_total_comments = HashMap::<usize, i32>::new();

    let mut author_dict = HashMap::<&str, HashMap<usize, i32>>::new();
    for i in 0..authors.len() {
        *author_dict.entry(authors[i]).or_default().entry(subreddits[i]).or_default() += counts[i];
        *sub_total_comments.entry(subreddits[i]).or_default() += counts[i];
    }

    let author_vec: Vec<HashMap<usize, i32>> = author_dict
        .into_values()
        .collect();
    (author_vec, sub_total_comments)
}

#[pyfunction]
fn build_network(authors: Vec<&str>, subreddits: Vec<usize>, counts: Vec<i32>) -> Vec<Vec<f32>> {
    let (author_vec, sub_total_comments) = make_author_vec(authors, subreddits, counts);
    compute_overlaps_arr(author_vec, sub_total_comments)
}

/// A Python module implemented in Rust.
#[pymodule]
fn network_builder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_network, m)?)?;
    Ok(())
}
