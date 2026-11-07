# Развертывание Viggle-Remake на RunPod Serverless (A100)
# Запускать в PowerShell, где уже задан $env:RUNPOD_API_KEY

# 0. Проверка ключа
if (-not $env:RUNPOD_API_KEY) {
    Write-Host "Ошибка: RUNPOD_API_KEY не задан" -ForegroundColor Red
    exit 1
}

# 1. Установи runpodctl (если ещё не установлен)
# npm install -g @runpod/runpodctl

# 2. Вход в аккаунт
# runpodctl auth login --api-key $env:RUNPOD_API_KEY

# 3. Создать Network Volume для весов (MiniMax-H3 ~144 GB, берём с запасом)
# runpodctl create network-volume --name viggle-weights --size 200

# 4. Собрать и запушить Docker-образ в ваш реестр
# docker build -t your-registry/viggle-remake:latest C:\Users\user\Desktop\dev\gpuapi\viggle-remake
# docker push your-registry/viggle-remake:latest

# 5. Создать Serverless Endpoint на A100
# min-workers = 0 — GPU выключается, когда нет задач (платите только за генерацию)
runpodctl create endpoint `
  --name viggle-remake-a100 `
  --image your-registry/viggle-remake:latest `
  --gpu-type "NVIDIA A100" `
  --gpu-count 1 `
  --min-workers 0 `
  --max-workers 2 `
  --idle-timeout 300 `
  --env BASE_MODEL_ID=MiniMaxAI/MiniMax-H3 `
  --env LORA_REPO_ID=Viggle/Viggle-Remake `
  --env HF_TOKEN=$env:HF_TOKEN `
  --env WEIGHTS_DIR=/workspace/weights `
  --env OUTPUT_DIR=/workspace/output `
  --volume viggle-weights:/workspace/weights

# 6. Получить ID эндпоинта
# runpodctl get endpoint --name viggle-remake-a100

# 7. Пример запроса на генерацию
# ENDPOINT_ID=xxx
# $body = @{
#   input = @{
#     model = "viggle-remake"
#     mode = "swap"
#     video = "https://.../source.mp4"
#     prompt = "make it snow"
#     refs = @("https://.../ref1.png")
#     pins = @(12)
#     frames = 124
#     short_edge = 480
#     seed = 0
#   }
# } | ConvertTo-Json
# Invoke-RestMethod -Uri "https://api.runpod.ai/v2/$ENDPOINT_ID/run" `
#   -Method POST `
#   -Headers @{Authorization = "Bearer $env:RUNPOD_API_KEY"; "Content-Type" = "application/json"} `
#   -Body $body

# Статус задачи:
# GET https://api.runpod.ai/v2/$ENDPOINT_ID/status/<TASK_ID>
