H265() {
ffmpeg -y -pix_fmt yuv420p -s $3*$4 -r $5 -i $1 -c:v libx265 -preset veryfast -tune zerolatency -x265-params "crf=${6}:keyint=${7}" -psnr -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug $2 2>&1 | grep 'Global' | tail -1
}

echo "H265 CRF = ${6}"
H265 $1 $2 $3 $4 $5 $6 $7
