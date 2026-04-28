"""Demo test of the complete multi-source analysis pipeline."""
import asyncio
from agents.problem_decoder import decode_problem
from agents.llm import OllamaLLM
from tools.data import MultiSourceFetcher
from tools.analysis_pipeline import (
    AnalysisPipeline,
    FetchStep,
    PreprocessStep,
    StatisticalTestStep,
    VisualizationStep,
    PipelineContext,
)
from tools.report import ReportBuilder
import json


async def test_problem_decoder():
    """Test problem decoding."""
    print("\n" + "="*60)
    print("TEST 1: Problem Decoder")
    print("="*60)
    
    from agents.llm import LLMConfig
    
    cfg = LLMConfig(
        model="qwen2.5:7b",
        host="http://localhost:11434",
    )
    llm = OllamaLLM(cfg)
    
    task = """Analyze the S&P 500 and tech sector performance over the last 2 years.
    Compare volatility patterns, compute rolling correlations, and identify
    trend reversals using statistical tests."""
    
    try:
        decoding = await decode_problem(llm, task)
        print(f"✓ Decoding successful")
        print(f"  Task Type: {decoding.task_type}")
        print(f"  Confidence: {decoding.confidence}")
        print(f"  Data Requirements: {len(decoding.data_requirements)} items")
        for req in decoding.data_requirements:
            print(f"    - {req.symbol} from {', '.join(req.sources)}")
        return decoding
    except Exception as e:
        print(f"✗ Decoding failed: {e}")
        return None


async def test_data_fetching(decoding):
    """Test multi-source data fetching."""
    print("\n" + "="*60)
    print("TEST 2: Multi-Source Data Fetching")
    print("="*60)
    
    fetcher = MultiSourceFetcher()
    
    # Extract symbols from decoding
    symbols = [req.symbol for req in decoding.data_requirements] if decoding else ["SPY", "QQQ"]
    
    try:
        data = fetcher.fetch(symbols, "2024-01-01", "2026-04-27")
        print(f"✓ Data fetching successful")
        for symbol, df in data.items():
            print(f"  {symbol}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
        return data
    except Exception as e:
        print(f"✗ Data fetching failed: {e}")
        return None


async def test_pipeline_execution(data):
    """Test pipeline execution."""
    print("\n" + "="*60)
    print("TEST 3: Pipeline Execution")
    print("="*60)
    
    if not data:
        print("✗ No data available for pipeline")
        return None
    
    try:
        # Create pipeline with steps
        steps = [
            FetchStep(list(data.keys()), ["yahoo"], "2024-01-01", "2026-04-27"),
            PreprocessStep(["drop_na", "add_returns"]),
            StatisticalTestStep(["correlation"], list(data.keys())[:2] if len(data) > 1 else list(data.keys())),
            VisualizationStep(["line_plot"], list(data.keys())),
        ]
        
        pipeline = AnalysisPipeline(steps)
        
        # Create initial context with data
        context = PipelineContext(data=data)
        
        # Execute
        context = pipeline.execute(context)
        print(f"✓ Pipeline execution successful")
        print(f"  Data in context: {len(context.data)} symbols")
        print(f"  Artifacts generated: {len(context.artifacts)}")
        print(f"  Variables stored: {len(context.variables)}")
        
        # Print some results
        for key, value in list(context.variables.items())[:5]:
            print(f"    {key}: {str(value)[:50]}...")
        
        return context
    except Exception as e:
        print(f"✗ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_report_generation(context, decoding):
    """Test report generation."""
    print("\n" + "="*60)
    print("TEST 4: Report Generation")
    print("="*60)
    
    if not context or not decoding:
        print("✗ Missing context or decoding for report generation")
        return
    
    try:
        builder = ReportBuilder("output/test_report")
        result = builder.generate_report(context, decoding)
        print(f"✓ Report generation successful")
        print(f"  LaTeX: {result['latex_path']}")
        if result['pdf_path']:
            print(f"  PDF: {result['pdf_path']}")
        print(f"  Datasets exported: {len(result['dataset_paths'])} files")
        print(f"  Plots embedded: {result['plots_saved']}")
        print(f"  Sections: {result['sections']}")
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  MULTI-SOURCE ANALYSIS PIPELINE - SYSTEM TEST".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Test 1: Problem Decoder
    decoding = await test_problem_decoder()
    
    # Test 2: Data Fetching
    data = await test_data_fetching(decoding)
    
    # Test 3: Pipeline Execution
    context = await test_pipeline_execution(data)
    
    # Test 4: Report Generation
    await test_report_generation(context, decoding)
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
