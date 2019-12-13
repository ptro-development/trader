Improvements:
(/) print all records count
(off) graph gaps distribution
(off) run code by using PyPy to see if you can make it faster and how much ?
(/) write algorithm for better samples
- (/) based on variation

Samples improvements:
(/) sample should have attributes:
- (/) information about reliability, how many gaps, how many matches with gaps
-- (/) time of snapshots when matched
-- (/) trading amount at the time

  !!! Be careful with predictions when there is not enough trading going on,
      especially if you data for prediction was with gaps which had to be filled
      with some temporary data

Big questions:
- what is the right size of sample ?
-- to have quicker response than human
-- to match the gaps ?

- (/) are there samples sentences ? (is one sample appearing before another one ?)
-- (/) yes, success it was observed and relation were mapped

Next:
- (/) write code to mark sample as monotonous up, down and up & down type
-- (/) write code to store it on in records on drive to be later read
-- (/) the samples it matches does not seam to be in size 8
--- (/) yes they are not, usually much smaller
-- (/) clean code, get rid of unnecessary arguments
- (off) deal with data which is has got the same epoch and different id
- (/) scheduled task does not happen as scheduled because there is only one
        worker available for all, try workers per queue

Finish of read only daemon:
- (off*) figure out how to read command line parameters to celery
- (/) clean code of hard coded values
-- (/) experiment with configuration loading
- (/) write script which does collapsing based on provided period and
        integrate it with ingest to speed up testing of trading algorithms against
        old data
- (off*) document code
- (/) clean directory as much as possible
- (/) modify scripts to use instead of relative numbers epoch time
- (/) celery init does need to run for every worker try to get rid of it if
        possible

Daemon:
- (/) write re-sampling of incoming stream with sync to existing samples from library
- (/) write code to replay log stream as it was coming
-- (/) just for loop through the gathered file later will be connected to the
socket
-- (/) write daemon to read stream from socket and highlights matching from different
samples records

Trading:
-- (off) design first trading algorithm and let it run on testing data it was
generated from with consideration of:
--- trading amount
--- gaps in sample
--- ?

Recently:
- (/) do not ask for computation of the same events in expected_trades
- (/) remove trades which are above max_allowed_focus
- (/) make ingest faster, is there a lock which could be used

- (/) move libraries from libs to trader/libs do all code changes to
        accommodate move
-- (/) get rid of trader/event.py and revisit why is it used in first place

- (/) tackle issue with two events at the same time but different
        correlation, look into o files

- (off) time.time in trade.py needs to be replaced by something so it is
        possible to run against training / old data
- (off) only probably one trade at the time otherwise you could heat up market
        the first trade will lock

*Patterns*
- (/) extend facts relations with UP / DOWN / VARIATE info
- (/) look at fusion of Events and Fact or forming Events relations
-- is not going to happen as they are not necessary god fit for merge, model
   was added to show relation

- (off) answer why some trades do not end with 100 and look at growing
        percentage pattern
-- the correlation disappear ?

- (/) some patterns appear very late at 80 % why are they not spotted
         earlier?
--- because parson correlation does not come with 0.975 match sooner, first
matches seams to appear later after 50% of sample :(

- (off) design trading / selling for countinues events
--- (off) save trade in data structure
          class: Trade(first_event, second_event_1)
                first_event
                [second_event_1, second_event_2 ...]
            trade status: updated / active, aborted / inactive
--- (off) write method:trades_status_check function running every rescale
          period to update trades status and if threshold is reached alert is
          triggered which could lead to aborting of trade
--- (off) write function to get trade signature / second event
          which can be used to update progress of trade, last update

Daemon for receiving trade data runs:
- it receives entry and send it to an incoming queue in celery #Queue:trade_data# witch
  runs in #Product:Trader#
- #Product:Trader# has process witch re-plays data to #Queue:store_trade_data# witch
  takes care of writing data into log file for later use of data modeling
- data in #Queue:trade_data# is collapsed to be ready from sample library
  matching and generating events into #Queue:recognised_event# to be ready for
  trading strategies
