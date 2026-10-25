# Retry transient imports/server initialization only, before any benchmark call.
start_server() {
 local attempt i ready
 for attempt in 1 2 3; do
  "$@" > "$run/server.log" 2>&1 &
  server_pid=$!
  ready=0
  for ((i=0;i<180;i++)); do
   if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
   if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
   sleep 5
  done
  if [[ "$ready" == 1 ]]; then return 0; fi
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  cp "$run/server.log" "$run/server-startup-failure-$attempt.log"
  tail -12 "$run/server.log"
  sleep 5
 done
 echo 'Server failed all three startup attempts before benchmark generation.' >&2
 return 1
}
