"""
Simple Example: Using SD15FlowTrainer (TrainBase Version)

This shows the simplest way to use the TrainBase-compatible trainer.
"""
import torch

# Note: In your repo, these would be:
# from geovocab2.train.trainers.sd15_flow_trainer import SD15FlowTrainer, SD15FlowConfig
from sd15_flow_trainer import SD15FlowTrainer, SD15FlowConfig


def minimal_example():
    """Minimal training example."""
    print("=" * 80)
    print("MINIMAL EXAMPLE: SD15FlowTrainer with TrainBase")
    print("=" * 80)

    # 1. Create configuration
    cfg = SD15FlowConfig(
        run_name="minimal_test",
        epochs=1,
        batch_size=8,
        num_samples=100,  # Very small for testing
        save_every=1
    )

    # 2. Create trainer (follows TrainBase contract)
    trainer = SD15FlowTrainer(cfg)
    print(f"\n✓ Trainer created: {trainer}")

    # 3. Get trainer info
    print("\nTrainer Info:")
    info = trainer.info()
    for k, v in info.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

    # 4. Transfer to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Moving to device: {device}")
    trainer.to(device)

    # 5. Train (this will take a while on first run due to model downloads)
    print("\n" + "-" * 80)
    print("Starting training...")
    print("-" * 80)
    results = trainer.train(epochs=1)
    print(f"\n✓ Training complete: {results}")

    # 6. Validate
    print("\n" + "-" * 80)
    print("Running validation...")
    print("-" * 80)
    val_results = trainer.validate()
    print(f"\n✓ Validation complete: val_loss = {val_results['val_loss']:.6f}")

    # 7. Save checkpoint
    print("\n" + "-" * 80)
    print("Saving checkpoint...")
    print("-" * 80)
    trainer.save_checkpoint("minimal_test")
    print("✓ Checkpoint saved")

    print("\n" + "=" * 80)
    print("MINIMAL EXAMPLE COMPLETE!")
    print("=" * 80)


def custom_config_example():
    """Example with custom configuration."""
    print("\n" + "=" * 80)
    print("CUSTOM CONFIG EXAMPLE")
    print("=" * 80)

    # Create custom config
    cfg = SD15FlowConfig(
        # Run settings
        run_name="custom_experiment",
        out_dir="./my_runs",
        ckpt_dir="./my_checkpoints",

        # Training
        epochs=5,
        batch_size=16,
        num_samples=10000,
        lr=5e-5,

        # Loss weights
        global_flow_weight=1.0,
        block_penalty_weight=0.25,  # Increase David influence
        kd_weight=0.5,

        # David fusion parameters
        alpha_timestep=0.6,
        beta_pattern=0.3,
        delta_incoherence=0.3,

        # Architecture
        use_local_flow_heads=True,  # Enable local flow heads
        pooling="adaptive"  # Use adaptive pooling
    )

    trainer = SD15FlowTrainer(cfg)

    print("\nCustom Configuration:")
    info = trainer.info()
    print(f"  Loss weights: {info['loss_weights']}")
    print(f"  Pooling: {info['pooling']}")
    print(f"  Active blocks: {len(info['active_blocks'])} blocks")

    # You would then train as normal:
    # trainer.to("cuda")
    # trainer.train()


def ablation_example():
    """Example showing how to do ablation studies."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY EXAMPLE")
    print("=" * 80)

    from geovocab2.train.trainers.sd15_trainers.core.assistant_model import NullAssistant
    from geovocab2.train.trainers.trainer_core import LossComposer

    class NoDAVIDTrainer(SD15FlowTrainer):
        """Trainer without David guidance (ablation baseline)."""

        def _initialize_models(self):
            super()._initialize_models()
            # Replace David with null assistant
            print("  Using NullAssistant (no David)")
            self.assistant = NullAssistant()

        def get_loss(self):
            # Only flow + KD, no David penalty
            from sd15_flow_trainer import create_flow_loss, create_kd_loss

            composer = LossComposer()
            composer.register("flow", create_flow_loss(self.cfg), 1.0)
            composer.register("kd", create_kd_loss(self.cfg), 0.25)
            return composer

    # Compare two configurations
    print("\nBaseline (No David):")
    baseline_cfg = SD15FlowConfig(
        run_name="baseline_no_david",
        epochs=2,
        num_samples=1000
    )
    baseline = NoDAVIDTrainer(baseline_cfg)
    print(f"  Losses: flow, kd")

    print("\nWith David:")
    david_cfg = SD15FlowConfig(
        run_name="with_david",
        epochs=2,
        num_samples=1000,
        block_penalty_weight=0.125
    )
    with_david = SD15FlowTrainer(david_cfg)
    print(f"  Losses: flow, kd, david_penalty")

    print("\nYou would train both and compare results:")
    print("  baseline.to('cuda').train()")
    print("  with_david.to('cuda').train()")


def checkpoint_example():
    """Example showing checkpoint save/load."""
    print("\n" + "=" * 80)
    print("CHECKPOINT EXAMPLE")
    print("=" * 80)

    cfg = SD15FlowConfig(
        run_name="checkpoint_test",
        ckpt_dir="./test_checkpoints"
    )

    # Train and save
    print("\nScenario 1: Train and save")
    trainer1 = SD15FlowTrainer(cfg)
    # trainer1.to("cuda")
    # trainer1.train(epochs=5)
    # trainer1.save_checkpoint("epoch_5")
    print("  Would save after epoch 5")

    # Load and resume
    print("\nScenario 2: Load and resume")
    trainer2 = SD15FlowTrainer(cfg)
    # trainer2.to("cuda")
    # trainer2.load_checkpoint("epoch_5")
    # trainer2.train(epochs=5)  # Continue from epoch 5
    print("  Would load from epoch 5 and continue")

    # Load for inference
    print("\nScenario 3: Load for inference only")
    trainer3 = SD15FlowTrainer(cfg)
    # trainer3.to("cuda")
    # trainer3.load_checkpoint("final")
    # model = trainer3.get_model()
    # # Use model for inference
    print("  Would load final checkpoint for inference")


def trainbase_contract_validation():
    """Validate that trainer follows TrainBase contract."""
    print("\n" + "=" * 80)
    print("TRAINBASE CONTRACT VALIDATION")
    print("=" * 80)

    trainer = SD15FlowTrainer()

    # Check all required methods exist
    required_methods = [
        'get_model',
        'get_loss',
        'get_datasets',
        'train',
        'validate',
        'test',
        'to',
        'info'
    ]

    print("\nChecking TrainBase contract compliance:")
    all_present = True
    for method in required_methods:
        has_method = hasattr(trainer, method) and callable(getattr(trainer, method))
        status = "✓" if has_method else "✗"
        print(f"  {status} {method}()")
        all_present = all_present and has_method

    # Check attributes
    print("\nChecking TrainBase attributes:")
    print(f"  ✓ name: {trainer.name}")
    print(f"  ✓ uid: {trainer.uid}")

    if all_present:
        print("\n✓ TRAINBASE CONTRACT FULLY SATISFIED")
    else:
        print("\n✗ TRAINBASE CONTRACT NOT SATISFIED")

    return all_present


if __name__ == "__main__":
    # Run examples
    try:
        # 1. Validate contract
        trainbase_contract_validation()

        # 2. Show custom config
        custom_config_example()

        # 3. Show ablation
        ablation_example()

        # 4. Show checkpointing
        checkpoint_example()

        # 5. Run minimal example (only if you have SD1.5 + David downloaded)
        print("\n" + "=" * 80)
        print("To run the full training example, uncomment the line below.")
        print("Note: First run will download SD1.5 (~4GB) and David (~100MB)")
        print("=" * 80)
        # minimal_example()  # Uncomment to run

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()