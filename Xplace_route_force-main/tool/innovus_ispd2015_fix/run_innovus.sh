#!/bin/bash
# nohup ./run_innovus.sh > nohup.out 2>&1 &
NAME="2024-03-04-17:54:32"

# cp -r $XPLACE_HOME/data/raw/ispd2015_fix $XPLACE_HOME/tool/innovus_ispd2015_fix

cd $XPLACE_HOME/tool/innovus_ispd2015_fix
rm -rf ./ispd2015_fix_xplace_route
python update_placement_def.py $XPLACE_HOME/result/$NAME

cd $XPLACE_HOME/tool/innovus_ispd2015_fix/innovus_work
rm -rf innovus.*
innovus -stylus -init run_all_route_xplace_route.tcl

# cd $XPLACE_HOME/tool/innovus_ispd2015_fix
# python parse_log.py ./innovus_work/innovus.log