from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://0.0.0.0:8265")

kick_off_pytorch_benchmark = (
    # Clone ray. If ray is already present, don't clone again.
    "git clone https://github.com/pytorch/examples.git || true;"
    # Run the training code.
    "python -m pip install -r examples/mnist/requirements.txt || true;"

    "cd examples/mnist/ || true;"
    "python main.py"
    " --epoch=2 --save-model"
)


submission_id = client.submit_job(
    entrypoint=kick_off_pytorch_benchmark,
    job_id="foo"
)

print("Use the following command to follow this Job's logs:")
print(f"ray job logs '{submission_id}' --address http://0.0.0.0:8265 --follow")