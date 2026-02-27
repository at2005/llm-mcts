#!/bin/bash
redis-cli --scan --pattern 'correct_answer:*' | xargs -r redis-cli DEL
redis-cli --scan --pattern 'weights:meta:*' | xargs -r redis-cli DEL
redis-cli del weights:version_counter weights:latest_version
redis-cli del replay_buffer
