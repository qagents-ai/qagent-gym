$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Speech
$voiceSynth = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try {
    [Console]::WriteLine('{"ready":true}')
    while ($null -ne ($voiceLine = [Console]::ReadLine())) {
        $voiceBuffer = $null
        try {
            $voiceRequest = $voiceLine | ConvertFrom-Json
            if ($voiceRequest.language -eq 'en') { $voiceCulture = 'en-US' }
            elseif ($voiceRequest.language -eq 'el') { $voiceCulture = 'el-GR' }
            else { throw 'Unsupported language' }
            if ([string]::IsNullOrWhiteSpace($voiceRequest.text) -or $voiceRequest.text.Length -gt 600) {
                throw 'Speech text must contain 1-600 characters'
            }
            $voiceMatch = @($voiceSynth.GetInstalledVoices() | Where-Object {
                $_.Enabled -and $_.VoiceInfo.Culture.Name -eq $voiceCulture
            })
            if ($voiceMatch.Count -eq 0) { throw "No enabled voice for $voiceCulture" }
            $voiceSynth.SelectVoice($voiceMatch[0].VoiceInfo.Name)
            $voiceBuffer = [System.IO.MemoryStream]::new()
            $voiceSynth.SetOutputToWaveStream($voiceBuffer)
            $voiceSynth.Speak([string]$voiceRequest.text)
            $voiceSynth.SetOutputToNull()
            $voiceJson = @{
                wav = [Convert]::ToBase64String($voiceBuffer.ToArray())
                voice = $voiceSynth.Voice.Name
                culture = $voiceSynth.Voice.Culture.Name
            } | ConvertTo-Json -Compress
            [Console]::WriteLine($voiceJson)
        } catch {
            [Console]::WriteLine((@{ error = $_.Exception.Message } | ConvertTo-Json -Compress))
        } finally {
            if ($null -ne $voiceBuffer) { $voiceBuffer.Dispose() }
        }
    }
} finally {
    $voiceSynth.Dispose()
}
