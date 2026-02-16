
"""Main entry point for the IA-MPPN machine learning pipeline.

This module orchestrates the complete machine learning workflow including:
- Data loading and preprocessing
- Dataset creation and normalization
- Model training and hyperparameter optimization
- Model evaluation and visualization
"""

from typing import Dict

from pipeline.pipeline import Pipeline

# Configuration paths for all pipeline components
CONFIG_PATHS: Dict[str, str] = {
    "io_config": "",
    "dataset_config": "./run_configs/config_dataset.json",
    "trainer_config": "./run_configs/config_trainer.json",
    "eval_optim_config": "./run_configs/config_evaluation_optimization.json",
    "experiment_config": "./run_configs/config_experiment.json",
    "resnet_config": "./run_configs/config_model_resnet.json",
    "IA_ViT_config": "./run_configs/config_model_IA_ViT.json",
    "dataloader_config": "./run_configs/config_dataloader.json",
    "IA_MPPN_config": "./run_configs/config_model_IA_MPPN.json",
    "MPPN_config": "./run_configs/config_model_MPPN.json",
}

# Available XES file names
XES_FILE_NAMES = ["PermitLog", "InternationalDeclarations", "DomesticDeclarations"]

# Attribute lists for different datasets
ATTR_LISTS = {
    "InternationalDeclarations": [
        "case:Amount",
        "concept:name",
        "org:resource",
        "org:role",
    ],
    "PermitLog": [
        "case:OrganizationalEntity",
        "case:OverspentAmount",
        "case:RequestedBudget",
        "org:resource",
        "concept:name",
    ],
    "DomesticDeclarations": [
        "case:Amount",
        "case:BudgetNumber",
        "org:resource",
        "concept:name",
        "org:role",
    ],
}


def create_pipeline(config_paths: Dict[str, str]) -> Pipeline:
    """Create and initialize the pipeline.
    
    Args:
        config_paths: Dictionary mapping config names to file paths.
        
    Returns:
        Initialized Pipeline instance.
    """
    return Pipeline(config_paths=config_paths)


def execute_pipeline(
    pipeline: Pipeline,
    read_xes: bool,
    xes_file_name: str,
    attrs_list: list[str],
    create_hdf5: bool,
    load_dataset: bool,
    optimize: bool,
    train_model: bool,
    evaluate: bool,
) -> None:
    """Execute the pipeline with the specified configuration.
    
    Args:
        pipeline: Pipeline instance to execute.
        read_xes: Whether to read XES file.
        synth_data: Whether to create synthetic data.
        xes_file_name: Name of the XES file to read.
        attrs_list: List of attributes to include.
        create_hdf5: Whether to create HDF5 dataset.
        load_dataset: Whether to load dataset.
        optimize: Whether to perform hyperparameter optimization.
        train_model: Whether to train the model.
        evaluate: Whether to evaluate the model.
        
    Raises:
        AssertionError: If both read_xes and synth_data are True.
    """
    
    # Step 1: Load or generate data
    if read_xes:
        pipeline.read(xes_file_name)
    
    # Step 2: Create HDF5 dataset
    if create_hdf5:
        pipeline.create_data(attrs_list=attrs_list)
    
    # Step 3: Load dataset
    if load_dataset:
        pipeline.load_dataset()
    
    # Step 4: Hyperparameter optimization
    if optimize:
        pipeline.analyze_hyperparameters()
    
    # Step 5: Train model
    if train_model:
        pipeline.train_model()
    
    # Step 6: Evaluate model
    if evaluate:
        pipeline.evaluate_model(interpre=False, vis=False, per=True)


def main() -> None:
    """Main entry point for the pipeline."""
    # Configuration
    xes_file_name = "InternationalDeclarations"
    attrs_list = ATTR_LISTS["InternationalDeclarations"]
    
    # Pipeline execution flags
    read_xes = False
    synth_data = False
    create_hdf5 = False  # HDF5 creation also applies normalization
    load_dataset = False
    optimize = False
    train_model = False
    evaluate = False
    
    # Create pipeline
    pipeline = create_pipeline(CONFIG_PATHS)
    
    # Execute pipeline
    execute_pipeline(
        pipeline=pipeline,
        read_xes=read_xes,
        xes_file_name=xes_file_name,
        attrs_list=attrs_list,
        create_hdf5=create_hdf5,
        load_dataset=load_dataset,
        optimize=optimize,
        train_model=train_model,
        evaluate=evaluate,
    )


if __name__ == "__main__":
    main()

        
