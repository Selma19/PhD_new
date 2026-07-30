#!/bin/zsh

#!!!to run on the LOCAL computer!!!

remoteName="login-dbn01.hpc.gwdg.de"

# use rsync so that it knows what was already transfered
# explanation of options:
# -r: recursive
# -v: run verbosely
# -z: compress the data during the sync (transport the data in compressed mode)
# -u only copy files with a newer modification time (or size difference if the times are equal)
# --delete: delete the files in target folder that do not exist in the source
rsync -rvuz --delete u24849@${remoteName}:${cidbnROOT}backup/ ./backup
