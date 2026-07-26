#!/bin/zsh

remoteName="login-dbn01.hpc.gwdg.de"

# use rsync so that it knows what was already transfered
rsync -chavzP --stats u24849@${remoteName}:${cidbnROOT}data/solo/stimulus.db ./data/solo/stimulus.db
