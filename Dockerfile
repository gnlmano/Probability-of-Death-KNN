FROM ubuntu:latest
LABEL authors="gnlm"

ENTRYPOINT ["top", "-b"]