from diagrams import Diagram
from diagrams.aws.storage import S3
from diagrams.aws.integration import StepFunctions
from diagrams.aws.ml import SagemakerTrainingJob
from diagrams.aws.management import Cloudwatch, Chatbot
from diagrams.saas.chat import Slack
from diagrams.aws.compute import Lambda as AwsLambda
from diagrams.aws.analytics import Quicksight
from diagrams.onprem.client import User

with Diagram(
    "AWS ML Pipeline (Screenshot Match)",
    filename="aws_arch_diagram",
    show=False,
    direction="TB"  # Top-to-Bottom layout
):

    # --- Top row ---
    api_log = S3("APIログ")
    shaped_data = S3("成型済みデータ")
    sf_model_build = StepFunctions("Step Functions\n(モデル構築用 ステートマシン)")
    training_data = S3("トレーニングデータ\n(教師あり)")
    training_job = SagemakerTrainingJob("トレーニングジョブ")
    trained_model = S3("学習済みモデル")

    cloudwatch = Cloudwatch("CloudWatch")
    chatbot = Chatbot("Chatbot")
    slack_notif = Slack("エラー発生時のSlack通知")

    # Connections for the top row
    api_log >> shaped_data >> sf_model_build
    sf_model_build >> training_data
    sf_model_build >> training_job
    sf_model_build >> trained_model

    # Error notifications (top SF → CloudWatch → Chatbot → Slack)
    sf_model_build >> cloudwatch >> chatbot >> slack_notif

    # --- Bottom row ---
    sf_inference = StepFunctions("Step Functions\n(推論用 ステートマシン)")
    inference_input = S3("推論対象データ")
    inference_job = SagemakerTrainingJob("推論バッチジョブ")
    inference_output = S3("推論結果")
    merge_lambda = AwsLambda("推論結果マージ用Lambda")
    merged_result = S3("推論結果\nマージ済みデータ")
    bi_tool = Quicksight("BIツール")
    user = User("ユーザー")

    # Connections for the bottom row
    inference_input >> sf_inference >> inference_job >> inference_output
    inference_job >> merge_lambda >> merged_result
    merged_result >> bi_tool >> user

    # Error notifications (bottom SF → CloudWatch → Chatbot → Slack)
    sf_inference >> cloudwatch

    # Also flow from “成型済みデータ” to the 推論用 Step Functions (as in screenshot)
    shaped_data >> sf_inference
