def test_artifact_directory_uses_runs_relative_external_path(tmp_path, monkeypatch):
    from sonar_encoder.training import artifacts

    run_root = tmp_path / "runs"
    artifact_root = tmp_path / "training_artifacts"
    monkeypatch.setattr(artifacts, "RUN_ROOT", run_root)
    monkeypatch.setattr(artifacts, "ARTIFACT_ROOT", artifact_root)
    local = run_root / "readout/uatd_future/cache"

    artifacts.create_artifact_directory(local)

    assert local.is_symlink()
    assert local.readlink() == artifact_root / "readout/uatd_future/cache"
    assert local.resolve() == artifact_root / "readout/uatd_future/cache"
    assert local.resolve().is_dir()
