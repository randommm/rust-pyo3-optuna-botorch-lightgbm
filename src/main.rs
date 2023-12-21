use lightgbm3::{Booster, Dataset};
use polars::prelude::*;
use pyo3::prelude::*;
use rand_distr::{Distribution, Normal};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let main_optuna = PyModule::from_code(
            py,
            r#"
import optuna
sampler = optuna.integration.BoTorchSampler(
    n_startup_trials=10,
)
study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    study_name='example-study',
    storage='sqlite:///optuna.db',
    load_if_exists=True,
)
        "#,
            "main_optuna.py",
            "main_optuna",
        )?;
        let study = main_optuna.getattr("study")?;
        // create folder models if not exist
        std::fs::create_dir_all("models")?;

        for _ in 0..10 {
            let trial = study.call_method0("ask")?;
            let trial_number = trial.getattr("number")?;
            let num_iterations: i64 = trial
                .call_method1("suggest_int", ("num_iterations", 100, 1000))?
                .extract()?;

            let rows = 1000;

            let mut rng = rand::thread_rng();
            let normal = Normal::new(0.0, 1.0).unwrap();

            let mut df = DataFrame::empty();

            let mut column = Vec::with_capacity(rows);
            for _ in 0..rows {
                column.push((normal.sample(&mut rng) as f64).sin());
            }
            df.with_column(Series::new("feature1", column)).unwrap();

            let mut column = Vec::with_capacity(rows);
            for _ in 0..rows {
                column.push((normal.sample(&mut rng) as f64).cos());
            }
            df.with_column(Series::new("feature2", column)).unwrap();

            let mut column = Vec::with_capacity(rows);
            for i in 0..rows {
                let val: f64 = 0.8 * normal.sample(&mut rng)
                    - 1.1
                        * df.column("feature1")
                            .unwrap()
                            .get(i)
                            .unwrap()
                            .try_extract::<f64>()
                            .unwrap()
                    + 1.1
                        * df.column("feature2")
                            .unwrap()
                            .get(i)
                            .unwrap()
                            .try_extract::<f64>()
                            .unwrap();
                let val = if val > 0.0 { 1.0 } else { 0.0 };
                column.push(val);
            }
            df.with_column(Series::new("groundt", column)).unwrap();

            let df_train = df.slice(0, 9 * rows / 10);
            let df_val = df.slice(9 * rows as i64 / 10, rows);

            let dataset = Dataset::from_dataframe(df_train, "groundt").unwrap();
            let params = json! {
               {
                    "num_iterations": num_iterations,
                    "objective": "binary",
                    "metric": "auc",
                }
            };

            let bst = Booster::train(dataset, &params).unwrap();

            // convert df_val to vec_of_vec
            let mut df_val_vec_of_vecs: Vec<Vec<f64>> = Vec::new();
            for row in df_val.iter().take(2) {
                let row_vec: Vec<f64> = row
                    .iter()
                    .map(|s| s.try_extract::<f64>().unwrap())
                    .collect();
                df_val_vec_of_vecs.push(row_vec);
            }

            let predictions = bst
                .predict_from_vec_of_vec(df_val_vec_of_vecs, false)
                .unwrap();
            //println!("predictions: {:?}", predictions);
            let mut loss = 0.0;
            for (i, prediction) in predictions.into_iter().enumerate() {
                let groundt = df_val
                    .column("groundt")
                    .unwrap()
                    .get(i)
                    .unwrap()
                    .try_extract::<usize>()
                    .unwrap();
                if groundt == 1 {
                    loss -= prediction[0].ln();
                } else {
                    loss -= (1. - prediction[0]).ln();
                }
            }
            loss /= rows as f64;
            println!("loss: {}", loss);
            study.call_method1("tell", (trial, loss))?;
            bst.save_file(format!("models/{trial_number}.lgb").as_str())
                .unwrap();
        }
        let best_trial = study.getattr("best_trial")?;
        println!("Best trial: {:?}", best_trial);

        Ok::<(), Box<dyn std::error::Error>>(())
    })?;
    Ok(())
}
