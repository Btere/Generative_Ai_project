pipeline {
  agent any

  options {
    timestamps()
    ansiColor('xterm')
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '20'))
  }

  parameters {
    choice(name: 'ENVIRONMENT', choices: ['dev', 'staging', 'prod'], description: 'Target environment for deployment')
    booleanParam(name: 'DEPLOY', defaultValue: false, description: 'Run deploy stage')
    string(name: 'DOCKER_IMAGE', defaultValue: '', description: 'Docker image name to build/push (e.g., myorg/myapp). Leave empty to skip Docker stage')
  }

  environment {
    PIP_DISABLE_PIP_VERSION_CHECK = '1'
    PIP_CACHE_DIR = '.pip-cache'
    PYTHONUNBUFFERED = '1'
    VENV = '.venv'
    PIP = '.venv/bin/pip'
    PYTHON = '.venv/bin/python'
    PYTEST = '.venv/bin/pytest'
    FLAKE8 = '.venv/bin/flake8'
    BLACK = '.venv/bin/black'
    BANDIT = '.venv/bin/bandit'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Setup Python') {
      steps {
        sh '''
set -euxo pipefail
python3 --version || true
python3 -m venv "${VENV}"
"${PYTHON}" -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  "${PIP}" install -r requirements.txt
fi
# CI tooling
"${PIP}" install --upgrade pytest pytest-cov flake8 black bandit build
'''
      }
    }

    stage('Lint') {
      steps {
        sh '''
set -euxo pipefail
# Linting (non-blocking suggestions can be made blocking by removing `|| true`)
"${FLAKE8}" || true
"${BLACK}" --check . || true
"${BANDIT}" -q -r . -x tests || true
'''
      }
    }

    stage('Test') {
      steps {
        sh '''
set -euxo pipefail
mkdir -p reports
if find . -type f \( -name 'test_*.py' -o -name '*_test.py' \) | grep -q .; then
  "${PYTEST}" -q \
    --junitxml=reports/junit.xml \
    --cov=. \
    --cov-report=xml:reports/coverage.xml \
    --cov-report=term-missing
else
  echo 'No tests found. Skipping.'
fi
'''
      }
    }

    stage('Package (Wheel)') {
      when {
        anyOf {
          expression { fileExists('pyproject.toml') }
          expression { fileExists('setup.cfg') }
          expression { fileExists('setup.py') }
        }
      }
      steps {
        sh '''
set -euxo pipefail
"${PYTHON}" -m build --wheel --outdir dist
'''
      }
    }

    stage('Docker Build & Push') {
      when {
        allOf {
          expression { fileExists('Dockerfile') }
          expression { return params.DOCKER_IMAGE?.trim() }
        }
      }
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
          sh '''
set -euxo pipefail
# Ensure Docker is available on this agent
command -v docker
IMAGE="${DOCKER_IMAGE}"
BRANCH_REF="${BRANCH_NAME:-${GIT_BRANCH:-unknown}}"
TAG_SUFFIX="$(echo "$BRANCH_REF" | sed -E 's#.*/##' | tr '/:' '-')-${BUILD_NUMBER}"
FULL_IMAGE="${IMAGE}:${TAG_SUFFIX}"

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

docker build -t "$FULL_IMAGE" .
docker push "$FULL_IMAGE"
echo "Pushed $FULL_IMAGE"
'''
        }
      }
    }

    stage('Deploy') {
      when {
        allOf {
          expression { return params.DEPLOY }
          anyOf { branch 'main'; branch 'master' }
        }
      }
      steps {
        sh '''
set -euxo pipefail
echo "Deploying to ${ENVIRONMENT}..."
# TODO: Add deployment commands here, e.g.:
# kubectl apply -f k8s/${ENVIRONMENT}/
# helm upgrade --install app chart/ --namespace ${ENVIRONMENT}
# serverless deploy --stage ${ENVIRONMENT}
'''
      }
    }
  }

  post {
    always {
      junit allowEmptyResults: true, testResults: 'reports/junit.xml'
      archiveArtifacts allowEmptyArchive: true, artifacts: 'reports/**/*, dist/**/*'
      deleteDir()
    }
  }
}