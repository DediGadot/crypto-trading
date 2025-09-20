"""Command Line Interface for SAFLA Trading System.

Provides a CLI for managing and monitoring the SAFLA trading system,
including training, prediction, analysis, and system health monitoring.
"""

import click
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint

from .config import get_config
from .memory import MemoryManager
from .core import NeuralCoordinator, FeedbackType
from .core.feedback_loops import FeedbackSignal

# Setup rich console
console = Console()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """SAFLA Cryptocurrency Trading System CLI."""
    ctx.ensure_object(dict)
    if config:
        ctx.obj['config_path'] = config

    # Display welcome message
    console.print(Panel(
        "[bold blue]SAFLA Cryptocurrency Trading System[/bold blue]\n"
        "[dim]Self-Aware Feedback Loop Algorithm for Adaptive Trading[/dim]",
        title="Welcome",
        style="blue"
    ))


@cli.command()
@click.option('--input-dim', default=64, help='Input dimension for neural network')
@click.option('--output-dim', default=8, help='Output dimension for neural network')
@click.option('--with-memory', is_flag=True, help='Initialize with memory system')
@click.pass_context
def init(ctx, input_dim, output_dim, with_memory):
    """Initialize SAFLA trading system."""
    try:
        console.print("[yellow]Initializing SAFLA trading system...[/yellow]")

        # Initialize memory manager if requested
        memory_manager = None
        if with_memory:
            console.print("  • Initializing memory systems...")
            memory_manager = MemoryManager()

        # Initialize neural coordinator
        console.print("  • Initializing neural coordinator...")
        coordinator = NeuralCoordinator(
            input_dim=input_dim,
            output_dim=output_dim,
            memory_manager=memory_manager
        )

        # Store in context
        ctx.obj['coordinator'] = coordinator
        ctx.obj['memory_manager'] = memory_manager

        console.print("[green]✓ SAFLA system initialized successfully![/green]")

        # Display system info
        _display_system_info(coordinator, memory_manager)

    except Exception as e:
        console.print(f"[red]Error initializing system: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--samples', default=100, help='Number of training samples to generate')
@click.option('--epochs', default=10, help='Number of training epochs')
@click.option('--batch-size', default=32, help='Training batch size')
@click.pass_context
def train(ctx, samples, epochs, batch_size):
    """Train the SAFLA neural network."""
    coordinator = ctx.obj.get('coordinator')
    if not coordinator:
        raise click.ClickException("System not initialized. Run 'init' command first.")

    try:
        console.print(f"[yellow]Training SAFLA network for {epochs} epochs...[/yellow]")

        # Generate synthetic training data
        console.print("  • Generating training data...")
        train_data = _generate_training_data(samples, coordinator.input_dim, coordinator.output_dim)

        # Training loop
        for epoch in track(range(epochs), description="Training epochs"):
            epoch_losses = []
            epoch_accuracies = []

            # Batch training
            for i in range(0, len(train_data['inputs']), batch_size):
                batch_inputs = train_data['inputs'][i:i+batch_size]
                batch_targets = train_data['targets'][i:i+batch_size]

                batch_data = {
                    'inputs': batch_inputs,
                    'targets': batch_targets
                }

                # Perform training step
                metrics = coordinator.train_step(batch_data, {'epoch': epoch, 'batch': i//batch_size})
                epoch_losses.append(metrics['total_loss'])
                epoch_accuracies.append(metrics['accuracy'])

            # Display epoch results
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)

            if epoch % 5 == 0 or epoch == epochs - 1:
                console.print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")

            # Provide feedback based on performance
            feedback_value = avg_accuracy
            coordinator.provide_feedback(
                FeedbackType.PERFORMANCE,
                feedback_value,
                metadata={'epoch': epoch, 'loss': avg_loss},
                context={'training': True}
            )

        console.print("[green]✓ Training completed successfully![/green]")

        # Show final performance
        _display_training_results(coordinator)

    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--samples', default=10, help='Number of predictions to make')
@click.option('--show-confidence', is_flag=True, help='Show confidence scores')
@click.pass_context
def predict(ctx, samples, show_confidence):
    """Make predictions using the SAFLA network."""
    coordinator = ctx.obj.get('coordinator')
    if not coordinator:
        raise click.ClickException("System not initialized. Run 'init' command first.")

    try:
        console.print(f"[yellow]Making {samples} predictions...[/yellow]")

        # Generate test inputs
        test_inputs = torch.randn(samples, coordinator.input_dim)

        predictions = []
        confidences = []

        for i in range(samples):
            input_tensor = test_inputs[i:i+1]
            context = {'prediction_id': i, 'timestamp': datetime.now()}

            prediction, metadata = coordinator.predict(input_tensor, context)
            predictions.append(prediction.numpy())
            confidences.append(metadata['confidence'])

        # Display results
        _display_predictions(predictions, confidences if show_confidence else None)

        console.print("[green]✓ Predictions completed![/green]")

    except Exception as e:
        console.print(f"[red]Error during prediction: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
@click.pass_context
def analyze(ctx):
    """Analyze system performance and identify improvements."""
    coordinator = ctx.obj.get('coordinator')
    if not coordinator:
        raise click.ClickException("System not initialized. Run 'init' command first.")

    try:
        console.print("[yellow]Analyzing system performance...[/yellow]")

        # Perform analysis and improvement
        results = coordinator.analyze_and_improve()

        # Display analysis results
        _display_analysis_results(results)

        # Get learning insights
        console.print("\n[yellow]Generating learning insights...[/yellow]")
        insights = coordinator.get_learning_insights()

        _display_learning_insights(insights)

        console.print("[green]✓ Analysis completed![/green]")

    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
@click.pass_context
def status(ctx):
    """Show system health and status."""
    coordinator = ctx.obj.get('coordinator')
    memory_manager = ctx.obj.get('memory_manager')

    if not coordinator:
        raise click.ClickException("System not initialized. Run 'init' command first.")

    try:
        console.print("[yellow]Checking system status...[/yellow]")

        # Get system health
        health = coordinator.get_system_health()
        _display_system_health(health)

        # Get memory status if available
        if memory_manager:
            console.print("\n[yellow]Memory System Status:[/yellow]")
            memory_status = memory_manager.get_system_status()
            _display_memory_status(memory_status)

    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--feedback-type', type=click.Choice(['performance', 'error', 'success', 'market', 'user']),
              required=True, help='Type of feedback')
@click.option('--value', type=float, required=True, help='Feedback value (0.0 to 1.0)')
@click.option('--context', help='JSON context for feedback')
@click.pass_context
def feedback(ctx, feedback_type, value, context):
    """Provide feedback to the system."""
    coordinator = ctx.obj.get('coordinator')
    if not coordinator:
        raise click.ClickException("System not initialized. Run 'init' command first.")

    try:
        # Parse context
        context_dict = {}
        if context:
            context_dict = json.loads(context)

        # Map feedback type
        feedback_type_map = {
            'performance': FeedbackType.PERFORMANCE,
            'error': FeedbackType.ERROR,
            'success': FeedbackType.SUCCESS,
            'market': FeedbackType.MARKET,
            'user': FeedbackType.USER
        }

        # Provide feedback
        coordinator.provide_feedback(
            feedback_type_map[feedback_type],
            value,
            metadata={'source': 'cli'},
            context=context_dict
        )

        console.print(f"[green]✓ Feedback provided: {feedback_type} = {value}[/green]")

    except Exception as e:
        console.print(f"[red]Error providing feedback: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--duration', default=60, help='Demo duration in seconds')
@click.pass_context
def demo(ctx, duration):
    """Run a demonstration of the SAFLA system."""
    console.print("[yellow]Starting SAFLA system demonstration...[/yellow]")

    try:
        # Initialize system
        console.print("  • Initializing system...")
        memory_manager = MemoryManager()
        coordinator = NeuralCoordinator(input_dim=32, output_dim=4, memory_manager=memory_manager)

        # Store in context
        ctx.obj['coordinator'] = coordinator
        ctx.obj['memory_manager'] = memory_manager

        # Run demonstration
        _run_demonstration(coordinator, memory_manager, duration)

        console.print("[green]✓ Demonstration completed![/green]")

    except Exception as e:
        console.print(f"[red]Error during demonstration: {e}[/red]")
        raise click.ClickException(str(e))
    finally:
        # Cleanup
        if 'coordinator' in ctx.obj:
            ctx.obj['coordinator'].shutdown()
        if 'memory_manager' in ctx.obj:
            ctx.obj['memory_manager'].shutdown()


# Helper functions

def _generate_training_data(samples, input_dim, output_dim):
    """Generate synthetic training data."""
    inputs = torch.randn(samples, input_dim)
    # Create targets with some relationship to inputs for learning
    targets = torch.tanh(inputs @ torch.randn(input_dim, output_dim) + torch.randn(output_dim))
    return {'inputs': inputs, 'targets': targets}


def _display_system_info(coordinator, memory_manager):
    """Display system information."""
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("Neural Network", "✓ Initialized")
    table.add_row("Feedback System", "✓ Active")
    table.add_row("Self-Improvement", "✓ Running")

    if memory_manager:
        table.add_row("Memory System", "✓ Active")
    else:
        table.add_row("Memory System", "○ Disabled")

    console.print(table)


def _display_training_results(coordinator):
    """Display training results."""
    # Get recent performance
    recent_performance = list(coordinator.coordination_state['recent_performance'])

    if recent_performance:
        table = Table(title="Training Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        final_performance = recent_performance[-1]
        avg_performance = np.mean(recent_performance[-10:]) if len(recent_performance) >= 10 else final_performance

        table.add_row("Final Performance", f"{final_performance:.4f}")
        table.add_row("Average (Last 10)", f"{avg_performance:.4f}")
        table.add_row("Total Feedback Signals", str(len(coordinator.feedback_manager.feedback_buffer)))

        console.print(table)


def _display_predictions(predictions, confidences=None):
    """Display prediction results."""
    table = Table(title="Predictions")
    table.add_column("Sample", style="cyan")
    table.add_column("Prediction", style="green")

    if confidences:
        table.add_column("Confidence", style="yellow")

    for i, pred in enumerate(predictions):
        pred_str = f"[{', '.join([f'{x:.3f}' for x in pred.flatten()[:4]])}...]"

        if confidences:
            table.add_row(str(i), pred_str, f"{confidences[i]:.3f}")
        else:
            table.add_row(str(i), pred_str)

    console.print(table)


def _display_analysis_results(results):
    """Display analysis results."""
    table = Table(title="Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Improvement Candidates", str(results['improvement_candidates']))
    table.add_row("Implementations", str(results['implementations']))
    table.add_row("Successful Implementations", str(results['successful_implementations']))

    if results['completed_cycle']:
        table.add_row("Completed Cycle", results['completed_cycle'])

    console.print(table)


def _display_learning_insights(insights):
    """Display learning insights."""
    # Neural Network Insights
    if 'neural_network' in insights:
        nn_insights = insights['neural_network']
        table = Table(title="Neural Network Insights")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Parameters", str(nn_insights.get('parameters', 'N/A')))
        table.add_row("Learning Rate", str(nn_insights.get('learning_rate', 'N/A')))
        table.add_row("Adaptations", str(nn_insights.get('training_history', {}).get('adaptation_count', 0)))

        console.print(table)

    # Coordination Insights
    if 'coordination' in insights:
        coord_insights = insights['coordination']
        table = Table(title="Coordination Insights")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Adaptations", str(coord_insights.get('total_adaptations', 0)))
        table.add_row("Neural Health", coord_insights.get('neural_health_status', 'Unknown'))
        table.add_row("Performance Stability", f"{coord_insights.get('performance_stability', 0):.3f}")

        console.print(table)


def _display_system_health(health):
    """Display system health status."""
    table = Table(title="System Health")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    # Overall status
    overall_status = health.get('overall_status', 'unknown')
    status_color = 'green' if overall_status == 'healthy' else 'yellow' if overall_status == 'degraded' else 'red'
    table.add_row("Overall", f"[{status_color}]{overall_status}[/{status_color}]")

    # Neural network health
    neural_health = health.get('neural_network', {})
    neural_status = neural_health.get('status', 'unknown')
    status_color = 'green' if neural_status == 'healthy' else 'yellow' if neural_status == 'degraded' else 'red'
    table.add_row("Neural Network", f"[{status_color}]{neural_status}[/{status_color}]")

    # Feedback system
    feedback_status = health.get('feedback_system', {})
    active_cycle = feedback_status.get('current_cycle_active', False)
    table.add_row("Feedback System", "✓ Active" if active_cycle else "○ Idle")

    # Coordination
    coordination = health.get('coordination', {})
    thread_alive = coordination.get('coordination_thread_alive', False)
    table.add_row("Coordination", "✓ Running" if thread_alive else "○ Stopped")

    console.print(table)


def _display_memory_status(memory_status):
    """Display memory system status."""
    table = Table(title="Memory Systems")
    table.add_column("System", style="cyan")
    table.add_column("Entries", style="green")
    table.add_column("Compression", style="yellow")

    memory_systems = memory_status.get('memory_systems', {})

    for system_name, system_stats in memory_systems.items():
        size = system_stats.get('size', 0)
        compression = system_stats.get('compression_ratio', 0.0)
        table.add_row(system_name.title(), str(size), f"{compression:.2f}")

    console.print(table)


def _run_demonstration(coordinator, memory_manager, duration):
    """Run a comprehensive demonstration."""
    console.print(f"  • Running demonstration for {duration} seconds...")

    import time
    start_time = time.time()
    step = 0

    while time.time() - start_time < duration:
        # Simulate trading step
        step += 1

        # Generate market data
        market_data = torch.randn(1, 32)
        context = {
            'step': step,
            'timestamp': datetime.now(),
            'market_condition': np.random.choice(['bull', 'bear', 'sideways'])
        }

        # Make prediction
        prediction, metadata = coordinator.predict(market_data, context)

        # Simulate trading outcome
        outcome = {
            'profit': np.random.normal(0.01, 0.05),  # Random profit/loss
            'success': np.random.random() > 0.4      # 60% success rate
        }

        # Create trading experience
        experience = {
            'context': context,
            'action': 'trade',
            'outcome': outcome,
            'prediction': prediction.numpy().tolist(),
            'confidence': metadata['confidence']
        }

        # Store in memory
        memory_manager.create_comprehensive_memory(experience)

        # Provide feedback
        feedback_value = 0.8 if outcome['success'] else 0.3
        coordinator.provide_feedback(
            FeedbackType.PERFORMANCE,
            feedback_value,
            context=context
        )

        # Periodic training
        if step % 10 == 0:
            # Generate training batch
            batch_data = {
                'inputs': torch.randn(8, 32),
                'targets': torch.randn(8, 4)
            }
            coordinator.train_step(batch_data, context)

        # Periodic analysis
        if step % 20 == 0:
            coordinator.analyze_and_improve()
            console.print(f"    Step {step}: Analyzed and improved system")

        # Brief pause
        time.sleep(0.1)

    # Show final results
    console.print(f"\n  • Demonstration completed after {step} steps")

    # Display summary
    insights = coordinator.get_learning_insights()
    memory_status = memory_manager.get_system_status()

    table = Table(title="Demonstration Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Steps", str(step))
    table.add_row("Feedback Signals", str(len(coordinator.feedback_manager.feedback_buffer)))
    table.add_row("Memory Entries", str(sum(
        system['size'] for system in memory_status['memory_systems'].values()
    )))
    table.add_row("Learning Cycles", str(len(coordinator.feedback_manager.learning_cycles)))

    console.print(table)


if __name__ == '__main__':
    cli()