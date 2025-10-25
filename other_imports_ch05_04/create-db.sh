#!/bin/bash

rm -f trades.db
sqlite3 trades.db < schema.sql