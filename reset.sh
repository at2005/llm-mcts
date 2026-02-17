#!/bin/bash
redis-cli --scan --pattern 'correct_answer:*' | xargs -r redis-cli DEL
redis-cli del replay_buffer
