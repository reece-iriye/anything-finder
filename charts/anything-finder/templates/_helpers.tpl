{{/*
Common labels applied to every resource.
*/}}
{{- define "anything-finder.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Derive the PBF filename from the URL (last path segment).
Example: https://download.geofabrik.de/north-america/us/texas-latest.osm.pbf
       → texas-latest.osm.pbf
*/}}
{{- define "anything-finder.pbfFilename" -}}
{{- splitList "/" .Values.osm.pbfUrl | last -}}
{{- end }}
