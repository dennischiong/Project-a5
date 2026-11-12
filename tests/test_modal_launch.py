"""Tests for harvest/modal_launch.py and modal_app.py settings. No Modal account, no network, no credits."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harvest import modal_launch as ml  # noqa: E402


def fake_repo(tmp_path):
    repo = tmp_path / "a5-work" / "a5-adversarial-internals"
    for rel in ("analysis/lib.py", "harvest/ws3_generate.py", "harvest/ws3_gpu_jobs.sh", "seeds.py", "requirements.txt", ml.PAIRS,
                "data/private_other.csv", "harvest/out/pilot.parquet", "figures/a.png", "PROGRESS.md", "CLAUDE.md"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x")
    return repo


def test_modal_tokens_and_hf_token_must_all_be_present_and_non_empty():
    assert ml.missing_tokens(dict(MODAL_TOKEN_ID="a", MODAL_TOKEN_SECRET="b", HF_TOKEN="c")) == []
    assert ml.missing_tokens(dict(MODAL_TOKEN_ID="a", MODAL_TOKEN_SECRET="", HF_TOKEN="c")) == ["MODAL_TOKEN_SECRET"]
    assert ml.missing_tokens(dict(MODAL_TOKEN_ID="a", MODAL_TOKEN_SECRET="b")) == ["HF_TOKEN"]
    assert ml.missing_tokens({}) == list(ml.TOKEN_NAMES)


def test_only_the_workspace_line_of_token_info_is_kept():
    out = "(Using MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables)\nToken: ak-SECRETID\nWorkspace: mrg-team (ac-123)\nUser: x"
    assert ml.workspace_line(out) == "Workspace: mrg-team (ac-123)"
    assert ml.workspace_line("Token: ak-SECRETID") is None


def test_stage_uploads_only_code_and_the_pairs_table(tmp_path):
    repo = fake_repo(tmp_path)
    stage = tmp_path / "stage"
    assert ml.stage_repo(repo, stage) == sorted(["analysis/lib.py", "harvest/ws3_generate.py", "harvest/ws3_gpu_jobs.sh", "seeds.py",
                                                 "requirements.txt", ml.PAIRS])
    assert (stage / ml.PAIRS).exists()
    for left_behind in ("data/private_other.csv", "harvest/out/pilot.parquet", "figures/a.png", "PROGRESS.md", "CLAUDE.md"):
        assert not (stage / left_behind).exists()


def test_stage_stops_without_the_pairs_table(tmp_path):
    repo = fake_repo(tmp_path)
    (repo / ml.PAIRS).unlink()
    with pytest.raises(FileNotFoundError):
        ml.stage_repo(repo, tmp_path / "stage")


def test_download_keeps_replies_in_data_and_gemma_c1_out_of_the_analysed_folder(tmp_path):
    repo = fake_repo(tmp_path)
    plan = dict(ml.download_plan(repo))
    assert plan[f"{ml.REMOTE_REPO}/data/ws3_replies"] == repo / "data"
    assert plan[f"{ml.REMOTE_WORK}/ws3_judges"] == repo.parent
    gemma = plan[f"{ml.REMOTE_WORK}/ws3_harvest/{ml.GEMMA_C1}"]
    assert gemma.name == "ws3_modal_pending" and "ws3_harvest" not in gemma.parts
    for q in ml.QWEN_C1 + ml.FULL_C1:
        assert plan[f"{ml.REMOTE_WORK}/ws3_harvest/{q}"].name == "ws3_modal_pending"


def test_run_command_detaches_and_rejects_unknown_jobs():
    cmd = ml.run_command("minimum")
    assert cmd[1:] == ["-m", "modal", "run", "--detach", "harvest/modal_app.py::run_job", "--job", "minimum"]
    with pytest.raises(ValueError):
        ml.run_command("setup")


def test_spinner_and_progress_lines_are_hidden_but_job_lines_are_kept():
    hidden = ["⠙ Running (1/1 containers active)... View app at", "⠴ Loading images (1 containers initializing)...",
              "Capturing CUDA graphs (PIECEWISE): 4%|▍ | 2/51 [00:00<00:05, 8.44it/s]",
              "Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:09<00:00, 9.81s/it]", "   ", "⠸",
              "└── ✓ Created Function run_job.", "│   /content/drive/MyDrive/a5-work/harvest/modal_app.py"]
    kept = ["=== replies_mistral: 09:32:32 UTC", "chunk 1/8: 25 s | {'length': 422, 'stop': 78} | ETA 2.9 min",
            "speed: 4000 prompts in 206 s | 5251 prompt tok/s", "DONE: 4000 replies | {'ok': 3999, 'unparsed': 1}",
            "Hugging Face account: hasifhammani | can read 3 models", "RuntimeError: Could not find nvcc",
            "STOP: the token in HF_TOKEN belongs to x", "finished: minimum. Stop the machine if nothing else is queued",
            "✓ App completed. View run at"]
    assert not any(ml.keep_line(l) for l in hidden)
    assert [l for l in hidden if ml.keep_line(l)] == []
    assert [l for l in kept if not ml.keep_line(l)] == []


def test_run_filtered_streams_kept_lines_and_stops_on_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ml, "REPO", tmp_path)
    script = tmp_path / "job.py"
    script.write_text("import sys\nprint('\\u2819 Running (1/1 containers active)... View app at')\nprint('DONE: 4 prompts')\n"
                      "sys.exit(int(sys.argv[1]))\n")
    ml.run_filtered([sys.executable, str(script), "0"])
    out = capsys.readouterr().out
    assert "DONE: 4 prompts" in out and "Running (" not in out and "1 progress lines hidden" in out
    with pytest.raises(SystemExit, match="exit code 3"):
        ml.run_filtered([sys.executable, str(script), "3"])


def test_cleanup_only_touches_text_and_weights_never_the_whole_volume():
    assert set(ml.CLEANUP_PATHS) == {f"{ml.REMOTE_REPO}/data", "/hf-cache"}
    assert "/" not in ml.CLEANUP_PATHS and ml.REMOTE_WORK not in ml.CLEANUP_PATHS


def test_cleanup_needs_confirmation_and_replies_on_drive(tmp_path, monkeypatch):
    repo = fake_repo(tmp_path)
    monkeypatch.setattr(ml, "REPO", repo)
    monkeypatch.setenv("MODAL_TOKEN_ID", "id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "secret")
    monkeypatch.setenv("HF_TOKEN", "hf")
    calls = []
    monkeypatch.setattr(ml, "call", lambda cmd, **kw: calls.append(cmd) or type("R", (), dict(returncode=0))())
    with pytest.raises(SystemExit):
        ml.cleanup(confirmed=False)
    with pytest.raises(SystemExit):
        ml.cleanup(confirmed=True)     # replies are not on Drive yet
    assert calls == []
    (repo / "data" / "ws3_replies").mkdir()
    ml.cleanup(confirmed=True)
    assert [c[-2] for c in calls] == list(ml.CLEANUP_PATHS)


def test_commands_and_messages_never_contain_the_token_values(tmp_path, monkeypatch, capsys):
    repo = fake_repo(tmp_path)
    monkeypatch.setattr(ml, "REPO", repo)
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-THEID")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-THESECRET")
    monkeypatch.setenv("HF_TOKEN", "hf_THEHFTOKEN")
    calls = []

    def fake_call(cmd, **kw):
        calls.append(cmd)
        return type("R", (), dict(returncode=0, stdout="Token: ak-THEID\nWorkspace: mrg (ac-1)"))()
    monkeypatch.setattr(ml, "call", fake_call)
    ml.check()
    ml.upload()
    ml.download()
    text = capsys.readouterr().out + " ".join(" ".join(c) for c in calls)
    assert "THEID" not in text and "THESECRET" not in text and "THEHFTOKEN" not in text
    assert "Workspace: mrg" in text and "Nothing was spent" in text
    assert not any("secret" in c for c in calls)              # nothing is stored as a Modal secret


def test_check_stops_when_hf_token_is_missing_before_calling_modal(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "REPO", fake_repo(tmp_path))
    monkeypatch.setenv("MODAL_TOKEN_ID", "id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "secret")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls = []
    monkeypatch.setattr(ml, "call", lambda cmd, **kw: calls.append(cmd))
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        ml.check()
    assert calls == []


class FakeHf:
    """Hugging Face API stand-in. The real token appears in its errors, as the worst case for logging."""

    def __init__(self, user="hasifhammani", bad_token=False, blocked=()):
        self.user, self.bad_token, self.blocked, self.checked = user, bad_token, set(blocked), []

    def whoami(self):
        if self.bad_token:
            raise ValueError("invalid token hf_THEVALUE")
        return dict(name=self.user)

    def auth_check(self, repo):
        self.checked.append(repo)
        if repo in self.blocked:
            raise PermissionError("no access with hf_THEVALUE")


def test_hf_check_passes_only_for_the_expected_account_and_checks_every_model(capsys):
    ma = pytest.importorskip("harvest.modal_app")
    api = FakeHf()
    assert ma.check_hf_access(api) == "hasifhammani"
    assert api.checked == list(ma.HF_REPOS)
    out = capsys.readouterr().out
    assert "hasifhammani" in out and "allenai/wildguard" in out


def test_hf_check_stops_on_someone_elses_token_before_checking_models():
    ma = pytest.importorskip("harvest.modal_app")
    api = FakeHf(user="dylanlerhj")
    with pytest.raises(RuntimeError, match="belongs to dylanlerhj, not hasifhammani"):
        ma.check_hf_access(api)
    assert api.checked == []


def test_hf_check_names_every_blocked_model_and_never_repeats_the_token():
    ma = pytest.importorskip("harvest.modal_app")
    with pytest.raises(RuntimeError) as blocked:
        ma.check_hf_access(FakeHf(blocked=["allenai/wildguard", "google/gemma-2-9b-it"]))
    assert "allenai/wildguard" in str(blocked.value) and "google/gemma-2-9b-it" in str(blocked.value)
    assert "mistralai" not in str(blocked.value) and "THEVALUE" not in str(blocked.value)
    with pytest.raises(RuntimeError) as rejected:
        ma.check_hf_access(FakeHf(bad_token=True))
    assert "did not accept" in str(rejected.value) and "THEVALUE" not in str(rejected.value)


def test_hf_check_covers_the_models_the_jobs_really_load():
    ma = pytest.importorskip("harvest.modal_app")
    from harvest import ws3_harvest as wh
    from harvest import ws3_judge as wj
    assert set(ma.HF_REPOS) == {wh.MODELS["gemma2-9b"]["id"], wh.MODELS["mistral-7b"]["id"], wj.JUDGE_ID}


def test_launcher_and_app_agree_and_the_app_guards_cost():
    pytest.importorskip("modal")
    from harvest import modal_app as ma
    assert (ma.VOLUME_NAME, ma.GPU, ma.JOBS) == (ml.VOLUME, "L40S", ml.JOBS)
    assert "Secret.from_name" not in Path(ma.__file__).read_text()   # no named secret is read from the workspace
    assert ma.TIMEOUT_HOURS * (1.95 + 0.45) <= 20                      # one attempt stays inside the $20 ask
    assert ma.RUNNER == f"{ma.MOUNT}{ml.REMOTE_REPO}/harvest/ws3_gpu_jobs.sh"
    assert ma.WORK == f"{ma.MOUNT}{ml.REMOTE_WORK}"
    assert Path(ml.REPO, "harvest", "ws3_gpu_jobs.sh").exists()


def test_download_skips_harvest_folders_already_on_drive(tmp_path):
    repo = fake_repo(tmp_path)
    done = repo.parent / "ws3_harvest" / ml.QWEN_C1[0]
    done.mkdir(parents=True)
    (done / "meta.json").write_text("{}")
    assert ml.already_on_drive(f"{ml.REMOTE_WORK}/ws3_harvest/{ml.QWEN_C1[0]}", repo)
    assert not ml.already_on_drive(f"{ml.REMOTE_WORK}/ws3_harvest/{ml.FULL_C1[0]}", repo)
    assert not ml.already_on_drive(f"{ml.REMOTE_WORK}/ws3_judges", repo)
