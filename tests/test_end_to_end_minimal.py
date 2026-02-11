import pytest
import os
import sys
import shutil
from scripts.evaluate_end_to_end import main

def test_end_to_end_minimal():
    """Run minimal end-to-end pipeline and verify artifacts."""
    
    # Clean up any previous test artifacts
    test_dirs = ['data/cycles', 'data/feature_tables', 'data/embeddings', 
                  'models/encoder', 'models/rl', 'reports']
    
    # Run pipeline with --quick flag
    sys.argv = ['evaluate_end_to_end.py', '--quick']
    
    try:
        main()
    except SystemExit as e:
        if e.code != 0:
            pytest.fail(f"Pipeline exited with code {e.code}")
    
    # Verify artifacts exist
    assert os.path.exists('data/feature_tables/features.parquet'), "Features not generated"
    
    encoder_files = [f for f in os.listdir('models/encoder') if f.endswith('.pt')]
    assert len(encoder_files) > 0, "Encoder not saved"
    
    emb_files = [f for f in os.listdir('data/embeddings') if f.endswith('.parquet')]
    assert len(emb_files) > 0, "Embeddings not generated"
    
    # Verify reports
    report_files = [f for f in os.listdir('reports') if f.endswith('.json')]
    assert len(report_files) > 0, "Report not generated"
    
    # Verify metrics are reasonable
    import json
    with open(f"reports/{report_files[0]}", 'r') as f:
        report = json.load(f)
    
    assert 'metrics' in report, "Metrics missing from report"
    
    # Sanity check: precision should be >= 0
    if 'precision' in report['metrics']:
        assert report['metrics']['precision'] >= 0.0, "Invalid precision value"
        assert report['metrics']['precision'] <= 1.0, "Precision out of range"
    
    print("✓ All artifacts created successfully")
    print(f"✓ Metrics: {report['metrics']}")
