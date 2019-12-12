library(azuremlsdk)

ws = load_workspace_from_config()

ds = get_default_datastore(ws)

cluster_name = "aml-trainer-gpu"
compute_target = get_compute(workspace = ws, cluster_name = cluster_name)

est = estimator(source_directory = "./fasantia/AITour/R", 
                entry_script = "train.R", 
                script_params = list("--data" = ds$path('hr-employee-attrition/dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')),
                compute_target = compute_target,
                cran_packages = c("caret", "optparse", "data.table", "kernlab"))

experiment_name = "employee-attrition-model-R"
exp = experiment(workspace = ws, name = experiment_name)
run = submit_experiment(experiment = exp, config = est)

wait_for_run_completion(run, show_output = TRUE)
