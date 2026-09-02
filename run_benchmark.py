"""Main entry point — runs the full AMH benchmark and generates report."""
import os
import sys
from datetime import datetime

# Infrastructure
from infrastructure.bedrock_client import BedrockClient

# Agents
from agents.amh_agent import AMHAgent
from agents.traditional_agent import StatelessAgent, BufferAgent, NaiveVectorAgent

# Scenarios
from scenarios import ALL_SCENARIOS

# Benchmark harness
from benchmarks.harness import BenchmarkHarness

# Report generator
from reports.report_generator import generate_report


def main():
    """Run the full benchmark suite."""
    print("\n" + "="*80)
    print(" Agent Memory Hierarchy (AMH) Benchmark".center(80))
    print("="*80)
    print("\nInitializing...\n")

    # Get AWS config from environment or defaults
    region = os.environ.get("AWS_REGION", "us-east-1")
    embedding_model = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    llm_model = os.environ.get("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")

    print(f"Region: {region}")
    print(f"Embedding Model: {embedding_model}")
    print(f"LLM Model: {llm_model}")
    print()

    # Initialize Bedrock client
    bedrock = BedrockClient(region=region)

    # Initialize all agents
    agents = {
        "StatelessAgent": StatelessAgent(
            bedrock_client=bedrock,
            llm_model_id=llm_model
        ),
        "BufferAgent": BufferAgent(
            bedrock_client=bedrock,
            buffer_size=10,
            llm_model_id=llm_model
        ),
        "NaiveVectorAgent": NaiveVectorAgent(
            bedrock_client=bedrock,
            top_k=5,
            embedding_model_id=embedding_model,
            llm_model_id=llm_model
        ),
        "AMHAgent": AMHAgent(
            bedrock_client=bedrock,
            mrdf_profile="recency_dominant",
            embedding_model_id=embedding_model,
            llm_model_id=llm_model
        ),
    }

    print(f"Agents initialized: {', '.join(agents.keys())}")
    print(f"Scenarios loaded: {len(ALL_SCENARIOS)}")
    print(f"  - {', '.join(s['name'] for s in ALL_SCENARIOS)}")
    print()

    # Initialize harness
    harness = BenchmarkHarness(
        bedrock_client=bedrock,
        judge_model_id=llm_model,
        evaluate_quality=True  # Enable LLM-as-judge
    )

    # Run all benchmarks
    print("Starting benchmark run...\n")
    try:
        results = harness.run_all(
            agents=agents,
            scenarios=ALL_SCENARIOS,
            verbose=True
        )
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save raw results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_output = f"benchmark_results_{timestamp}.json"
    harness.save_results(json_output)

    # Generate HTML report
    print("\nGenerating HTML report...")
    html_output = f"benchmark_report_{timestamp}.html"
    report_path = generate_report(results, html_output)

    # Summary
    print("\n" + "="*80)
    print(" Benchmark Complete ".center(80))
    print("="*80)
    print(f"\nRaw data: {json_output}")
    print(f"HTML report: {report_path}")
    print(f"\nOpen the HTML file in your browser to view the full interactive report.")
    print("\nKey findings summary:")

    # Compute quick summary stats
    gains = results.gains_vs_baseline("StatelessAgent")
    amh_gains = gains.get("AMHAgent", {})

    if amh_gains:
        print("\nAMH vs Stateless Baseline (averaged across scenarios):")
        metrics = ["latency_delta_pct", "cost_delta_pct", "token_delta_pct",
                   "quality_delta_pct", "personalization_delta_pct"]
        for scenario, scenario_gains in amh_gains.items():
            print(f"\n  {scenario}:")
            for m in metrics:
                val = scenario_gains.get(m, 0)
                print(f"    {m.replace('_delta_pct', '').replace('_', ' ').title()}: {val:+.1f}%")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
