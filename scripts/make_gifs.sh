#!/usr/bin/env bash
# Build README preview GIFs from each agent's replay video on the Hugging Face Hub.
#   scripts/make_gifs.sh            build GIFs that do not exist yet
#   scripts/make_gifs.sh --force    rebuild everything
#   scripts/make_gifs.sh <name>...  only these agents (repo names, e.g. ppo-LunarLander-v1)
# Source: the model repo's replay video, downloaded anonymously (public repos, no token) into the gitignored .cache/mp4.
# Fallback: $DRIVE_DIR/<local path> only if the Hub repo has no video.
# GIF: 320 px wide, <=10 fps, 7 s from the end of short episodes (<=15 s) or the most active part of longer videos, two-pass palette (ffmpeg),
#      then gifsicle -O3 --lossy=80 if gifsicle is installed.
# Output: assets/gifs/<repo-name>.gif, plus assets/gifs/manifest.tsv (agent, source, duration, size).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OWNER=elliemci; WIDTH=320; FPS=10; LEN=7
CACHE=.cache/mp4; OUT=assets/gifs; PY="${PYTHON:-$( [ -x venv/bin/python ] && echo venv/bin/python || echo python3 )}"
FORCE=0; ONLY=()
for a in "$@"; do [ "$a" = "--force" ] && FORCE=1 || ONLY+=("$a"); done

# repo-name | video file in the repo | local fallback (relative to $DRIVE_DIR) or -
AGENTS=(
  "q-FrozenLake-v1-4x4-noSlippery|replay.mp4|q_learning/videos/frl-replay.mp4"
  "q-Taxi-v3-noRainy|replay.mp4|q_learning/videos/taxi-replay.mp4"
  "ppo-LunarLander-v2|replay.mp4|-"
  "ppo-LunarLander-v3|ppo-lunarlander-step_0_500.mp4|-"
  "dqn-SpaceInvadersNoFrameskip-v4|replay.mp4|-"
  "Reinforce-CartPole|replay.mp4|policy_gradient/replay.mp4"
  "Reinforce-PixelCopter|replay.mp4|policy_gradient/replay_seed123.mp4"
  "a2c-PandaReachDense-v3|replay.mp4|actor_critics/videos/a2c-PandaReachDense-v3-replay-step-0-to-step-500.mp4"
  "a2c-PandaPickAndPlace-v3|replay.mp4|actor_critics/videos/a2c-PandaPickAndPlace-v3-replay-step-0-to-step-500.mp4"
  "ppo-LunarLander-v1|replay.mp4|-"
  "doom_health_gathering_supreme|replay.mp4|-"
  "deadly_corridor_experiment|replay.mp4|-"
)
# Unity agents with no replay video yet (waiting for screen recordings in recordings/<repo-name>.mp4)
UNITY=(ppo-Huggy ppo-Snowball-Target ppo-Pyramids poca-SoccerTwos)

mkdir -p "$CACHE" "$OUT"; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
[ -f "$OUT/manifest.tsv" ] || printf 'agent\tsource\tstart_s\tgif_s\tsize_kb\n' > "$OUT/manifest.tsv"
have_gifsicle=0; command -v gifsicle >/dev/null && have_gifsicle=1

build() { # name src_video source_label
  local name="$1" src="$2" label="$3" gif="$OUT/$1.gif"
  local dur fps start len
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$src")
  fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$src" | awk -F/ '{printf "%d", ($2?$1/$2:$1)+0.5}')
  [ "$fps" -gt "$FPS" ] && fps=$FPS; [ "$fps" -lt 1 ] && fps=1
  # Short episode videos (<=15 s): take the LAST $LEN s, where the outcome (landing, goal) is. Longer videos: most active window.
  if awk -v d="$dur" -v l="$LEN" 'BEGIN{exit !(d<=15 && d>l)}'; then start=$(awk -v d="$dur" -v l="$LEN" 'BEGIN{printf "%.2f", d-l}')
  else start=$("$PY" scripts/_pick_window.py "$src" "$LEN" 2>/dev/null || echo 0); fi
  len=$(awk -v d="$dur" -v s="$start" -v l="$LEN" 'BEGIN{r=d-s; print (r<l?r:l)}')
  local vf="fps=$fps,scale=$WIDTH:-1:flags=lanczos"
  ffmpeg -v error -y -ss "$start" -t "$len" -i "$src" -vf "$vf,palettegen=stats_mode=diff" "$tmp/pal.png"
  ffmpeg -v error -y -ss "$start" -t "$len" -i "$src" -i "$tmp/pal.png" \
    -lavfi "$vf [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -loop 0 "$gif"
  [ $have_gifsicle = 1 ] && gifsicle -O3 --lossy=80 -b "$gif"
  printf '%s\t%s\t%s\t%.1f\t%d\n' "$name" "$label" "$start" "$len" "$(( $(stat -f%z "$gif") / 1024 ))" >> "$OUT/manifest.tsv"
  echo "ok   $name  ($label, ${len}s from ${start}s)"
}

for row in "${AGENTS[@]}"; do
  IFS='|' read -r name vfile local_path <<< "$row"
  if [ ${#ONLY[@]} -gt 0 ] && ! printf '%s\n' "${ONLY[@]}" | grep -qx "$name"; then continue; fi
  if [ -f "$OUT/$name.gif" ] && [ $FORCE = 0 ]; then echo "skip $name (exists; use --force)"; continue; fi
  [ $FORCE = 1 ] && sed -i '' "/^$name\t/d" "$OUT/manifest.tsv"
  mp4="$CACHE/$OWNER/$name/$vfile"; mkdir -p "$(dirname "$mp4")"
  if [ -s "$mp4" ] || curl -fsSL -o "$mp4" "https://huggingface.co/$OWNER/$name/resolve/main/$vfile"; then
    build "$name" "$mp4" HF
  elif [ "$local_path" != "-" ] && [ -n "${DRIVE_DIR:-}" ] && [ -f "$DRIVE_DIR/$local_path" ]; then
    rm -f "$mp4"; build "$name" "$DRIVE_DIR/$local_path" local
  else
    rm -f "$mp4"; echo "FAIL $name: no video on the Hub and no local fallback" >&2
  fi
done

for name in "${UNITY[@]}"; do
  if [ -f "recordings/$name.mp4" ] && { [ ! -f "$OUT/$name.gif" ] || [ $FORCE = 1 ]; }; then build "$name" "recordings/$name.mp4" recording
  elif [ ! -f "$OUT/$name.gif" ]; then echo "wait $name (waiting for screen recording: recordings/$name.mp4)"; fi
done
echo; column -t -s$'\t' "$OUT/manifest.tsv"; du -ch "$OUT"/*.gif | tail -1
