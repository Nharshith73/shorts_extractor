param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Channel,

    [Parameter(Mandatory=$false)]
    [switch]$Kick,

    [Parameter(Mandatory=$false)]
    [int]$Min = 5,

    [Parameter(Mandatory=$false)]
    [string]$Ai = "groq",

    [Parameter(Mandatory=$false)]
    [string]$Layout = "blurred_background"
)

$argsList = @($Channel, "--min", $Min, "--ai", $Ai, "--layout", $Layout)
if ($Kick) {
    $argsList += "--kick"
}

python -m app.live.watcher @argsList
