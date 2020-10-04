#!/bin/bash

for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            mode)              mode=${VALUE} ;;
            rebuild)    rebuild=${VALUE} ;;
            rm_bg_key)    rm_bg_key=${VALUE} ;;
            *)
    esac

done


docker-compose down
if [ "$rebuild" = "rebuild" ]; then
  echo "Let's rebuild image"
  docker-compose build
fi

if [ "$rm_bg_key" != "" ]; then
  echo "KEY IS OK"
fi

if [ "$mode" = "d" ]; then
  echo "Run container in background"
  docker-compose up -d
else
  echo "Run container in foreground"
  docker-compose up
fi



