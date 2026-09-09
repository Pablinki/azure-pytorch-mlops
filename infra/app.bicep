// Container app + AcrPull role assignment. CI redeploys this module with the real image tag.
param baseName string
param location string = resourceGroup().location
param acrName string
param acaEnvName string

@description('Full image reference. The default is a public placeholder used only on first provisioning.')
param apiImage string = 'mcr.microsoft.com/k8se/quickstart:latest'

param minReplicas int = 0 // scale to zero when idle
param maxReplicas int = 5
param concurrentRequests int = 20 // KEDA HTTP rule: add a replica per 20 concurrent requests

@description('False in CI: the assignment already exists and the CI identity should not need roleAssignments/write.')
param assignAcrPull bool = true

var isPlaceholder = apiImage == 'mcr.microsoft.com/k8se/quickstart:latest'

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = { name: acrName }
resource acaEnv 'Microsoft.App/managedEnvironments@2024-03-01' existing = { name: acaEnvName }

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${baseName}-api'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      ingress: { external: true, targetPort: isPlaceholder ? 80 : 8000, transport: 'auto' }
      registries: isPlaceholder ? [] : [{ server: acr.properties.loginServer, identity: 'system' }]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'LOG_FORMAT', value: 'json' }
            { name: 'TEXTCLF_HOST', value: '0.0.0.0' }
            { name: 'TEXTCLF_MODEL_URI', value: 'file:///app/model' }
            { name: 'TEXTCLF_TORCH_THREADS', value: '2' }
          ]
          probes: isPlaceholder
            ? []
            : [
                {
                  type: 'Liveness'
                  httpGet: { path: '/health', port: 8000 }
                  initialDelaySeconds: 10
                  periodSeconds: 30
                }
                {
                  type: 'Readiness'
                  httpGet: { path: '/ready', port: 8000 }
                  initialDelaySeconds: 5
                  periodSeconds: 10
                }
              ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: { metadata: { concurrentRequests: string(concurrentRequests) } }
          }
        ]
      }
    }
  }
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (assignAcrPull) {
  name: guid(acr.id, api.name, 'AcrPull')
  scope: acr
  properties: {
    // AcrPull built-in role
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d'
    )
    principalId: api.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output fqdn string = api.properties.configuration.ingress.fqdn
output principalId string = api.identity.principalId
