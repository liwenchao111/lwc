#!/bin/bash
# nohup ./run_innovus.sh > nohup.out 2>&1 &
NAME="2024-07-15-16:23:12"

# cp -r $XPLACE_HOME/data/raw/ispd2015_fix $XPLACE_HOME/tool/innovus_ispd2015_fix

cd /home/lwc/lwc/Xplace_route_force-main/tool/innovus_ispd2015_fix
rm -rf ./ispd2015_fix_xplace_route
python update_placement_def.py /home/lwc/lwc/Xplace_route_force-main/result/$NAME

cd /home/lwc/lwc/Xplace_route_force-main/tool/innovus_ispd2015_fix/innovus_work
rm -rf innovus.*
innovus -stylus -init run_all_route_xplace_route.tcl

# cd $XPLACE_HOME/tool/innovus_ispd2015_fix
# python parse_log.py ./innovus_work/innovus.log