extract() {
ffmpeg -i $1 $2 2>&1 | tail -2
}

extract $1 $2
