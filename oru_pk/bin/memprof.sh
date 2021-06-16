usage() { echo "Usage: memprof [-s RATE] [-l LIM] PROG [ARGS [ARGS...]]" ; } ;
set -e
SAMPLE_RATE=1

while [[ -n $1 ]]; do
  case "$1" in
    -s)
      shift
      SAMPLE_RATE=$1
      shift
      ;;
    -l)
      shift
      LIMIT=$((${1}*1000))
      shift
      ;;
    -*)
      usage
      ;;
    *)
      break
      ;;
  esac
done

if [[ -n LIMIT ]] ; then ulimit -S -m $LIMIT || usage ; fi



PROG=$1
shift
PROG_ARGS=$@

