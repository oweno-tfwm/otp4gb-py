"""Run OTP4GB-py and produce isochrones.

Requires OTP >= v2.2 which provides isochrone generation support

Tested against OTP 2.3

"""
from __future__ import annotations

import sys
import atexit
import csv
import datetime
from functools import partial
import glob
import logging
import math
import concurrent.futures
import os
import threading
import zipfile 
import pandas as pd
import itertools 

from otp4gb.centroids import ZoneCentroids, load_centroids, ZoneCentroidColumns
from otp4gb.config import ProcessConfig, load_config
from otp4gb.otp import Server
from otp4gb.util import Timer, chunker
from otp4gb import parameters, routing 
from otp4gb.batch import build_run_spec, setup_worker, build_run_spec, run_batch, run_batch_catch_errors  
from otp4gb.logging import configure_app_logging
from process import ProcessArgs, loadCentroids
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


_FILENAME_PATTERN : str = (
    "Buffered{buffer_size}m_IsochroneBy_{mode}_ToZone_"
    "{location_name}_To{arrive_depart}_{by_or_between}_{arrival_time}_"
    "within_{journey_time:_>4n}_mins.geojson"
)


class IsochroneJobExecutor:
    @abstractmethod
    def run(self, server :Server) -> int:
        raise TypeError("IsochroneJobExecutor: pure virtual method not implemented by derived class")


    def factory( outputDirectory :str, config :ProcessConfig ) -> IsochroneJobExecutor:

        #use default isochrone config settings if not supplied
        if config.isochrone_configuration is None :
            config.isochrone_configuration = parameters.IsochroneConfiguration()
            
        jobExector: IsochroneJobExecutor 
    
        if config.isochrone_configuration.union_all_times:
            jobExector = _IsochroneJobExecutorGroupByLocation(outputDirectory, config)
        else:
            jobExector = _IsochroneJobExecutorGroupByDate(outputDirectory, config)

        return jobExector
    

    def cleanBadCharsFromFilename( filename:str ) -> str:
        filename = filename.replace(',', '_')
        filename = filename.replace(':', '-')
        filename = filename.replace(' ', '_')
        return filename



class _IsochroneJobExecutorBase(IsochroneJobExecutor):

    _compressing_lock :threading.Lock = threading.Lock()
    _csv_lock :threading.Lock = threading.Lock()
    _csv_header_written :bool = False

    def __init__(self, outputFolder :str, config :ProcessConfig):
        self.outputFolder = outputFolder
        self.config = config

        #use default isochrone config settings if not supplied
        if self.config.isochrone_configuration is None :
            self.config.isochrone_configuration = parameters.IsochroneConfiguration()


    def run(self, server :Server) -> int:        

        errorCount = 0
        self._csv_header_written = False

        centroids :ZoneCentroids = loadCentroids(self.config)
        
        jobs :list() = self._buildJobList(server, centroids)

        jobs.sort(key=self._jobSorterFn)

        # Create output directory
        isochrones_dir = os.path.join(self.outputFolder, 'isochrones')
        if not os.path.exists(isochrones_dir):
            os.makedirs(isochrones_dir)

        #could probably remove this now moved from executing in seperate processes to threads instead
        setup_worker({
                'output_dir': isochrones_dir,
                'centroids': centroids,
                'buffer_size': self.config.isochrone_configuration.buffer_metres,
                'FILENAME_PATTERN': _FILENAME_PATTERN,
                'name_key': self.config.isochrone_configuration.zone_column,
            },)

        with concurrent.futures.ThreadPoolExecutor( max_workers=self.config.number_of_threads ) as threadPool:
        
            try:        
                errorCount = self._executeJobs( threadPool, isochrones_dir, jobs )
        
            except Exception as e:
                errorCount += 100
                err_msg = f"{e.__class__.__name__}: {e}"
                logger.error("T%d:Exception thrown by _executeJobs() - %s", threading.get_ident(), err_msg)

            logger.info("shutting down threadpool")
            threadPool.shutdown()

        return errorCount

    @abstractmethod
    def _jobSorterFn(self,item):
        pass

    @abstractmethod
    def _jobGrouperFn(self,item):
        pass

    @abstractmethod
    def _executeJobs(    self,
                            threadPool :concurrent.futures.ThreadPoolExecutor,
                            isochrones_dir :str,
                            jobList :list) -> int:   
        pass

    def _buildJobList(self, server :Server, centroids :ZoneCentroids) -> list():
        
        jobs = list()

        for time_period in self.config.time_periods:

            travel_datetime = datetime.datetime.combine(
                self.config.date, time_period.travel_time
            )
            
            # Assume time is in local timezone
            travel_datetime = travel_datetime.astimezone()
            logger.info(
                "Building job list for datetime (%s) : local timezone (%s) has been assumed",
                travel_datetime, travel_datetime.tzinfo,
            )
                
            jobs += build_run_spec(name_key=self.config.isochrone_configuration.zone_column, 
                                        modes=self.config.modes, 
                                        centroids=centroids, 
                                        arrive_by=travel_datetime,
                                        travel_time_max=time_period.search_window_minutes, 
                                        travel_time_step=self.config.isochrone_configuration.step_minutes,
                                        server=server,
                                        arrive=True)         
        return jobs


    def _compressFiles(self, directory :str, inputPattern :str, outputFilename:str ):
        
        # compress the output files so it's easier to handle
        outputFilename = IsochroneJobExecutor.cleanBadCharsFromFilename( outputFilename )

        added_files = []

        # Write list to csv in batches
        logger.info("T%d:Compressing files directory=(%s) pattern=(%s), output=(%s)", threading.get_ident(), directory, inputPattern, outputFilename)

        with self._compressing_lock:
        
            with zipfile.ZipFile(os.path.join(directory, outputFilename), 'w', compression=zipfile.ZIP_DEFLATED ) as zip_file:
                for file_path in glob.glob(os.path.join(directory, inputPattern)):
                    zip_file.write(file_path, os.path.basename(file_path) )
                    added_files.append(file_path)

            # Delete the successfully added files
            for file_path in added_files:
                os.remove(file_path)


    def _appendListToFile(self, filename: str, list :list ):

        # Write list to csv in batches
        logger.info("Writing %d rows to %s", len(list), filename)

        with self._csv_lock:
            with open(filename, 'a') as file:
                pd.DataFrame.from_dict(list).to_csv(file, index=False, header=not self._csv_header_written, lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
                self._csv_header_written=True



class _IsochroneJobExecutorGroupByDate(_IsochroneJobExecutorBase):
    def __init__(self, outputFolder :str, config :ProcessConfig):
        super().__init__(outputFolder, config)

    def _jobSorterFn(self,item):
        return (item["travel_time"], item["name"], item["mode"])

    def _jobGrouperFn(self,item):
        return (item["travel_time"])
        
    def _executeJobs(    self,
                            threadPool :concurrent.futures.ThreadPoolExecutor,
                            isochrones_dir :str,
                            jobs :list) -> int:   
        errorCount = 0        

        runDate = datetime.date.today().isoformat()
        
        for key, group in itertools.groupby(jobs, key=self._jobGrouperFn):    
            errorCount += self._runJobGroup(threadPool, isochrones_dir, key, list(group), runDate) 

        return errorCount

    def _runJobGroup(self, 
                            threadPool :concurrent.futures.ThreadPoolExecutor,
                            isochrones_dir :str,
                            dateString :str, 
                            jobs :list,
                            runDate :str) -> int:  

        travel_datetime = datetime.datetime.fromisoformat(dateString)
            
        logger.info( "Starting run for datetime (%s)", travel_datetime )

        matrix_filename = f"AreatoAreaTravelTimeMatrix_ToArriveBy_{travel_datetime.strftime('%Y%m%d_%H%M')}"
       
        errorCount = 0

        with open(os.path.join( self.outputFolder, matrix_filename ) + ".csv", 'w') as file:
            pass  # truncate the file and close it again
        
        self._csv_header_written = False
            
        numChunks = math.ceil(len(jobs) / (self.config.number_of_threads * 5))
            
        for idx, batch in enumerate(chunker(jobs, self.config.number_of_threads*5)):
            logger.info(
                "==================== Running batch %d of %d ====================", idx+1, numChunks)
            logger.info("Dispatching %d jobs", len(batch))

            jobsInList = [[job] for job in batch]
            taskFutures = [threadPool.submit(run_batch_catch_errors, job) for job in jobsInList] 

            matrix = list()
      
            for task in concurrent.futures.as_completed(taskFutures):
                taskResult = task.result()

                if taskResult is None:  
                    logger.error("null taskResult object recieved")
                elif "error" in taskResult:
                    errorCount += 1 #message logged in the execution thread
                else:
                    # This is a list comprehension which flattens the results
                    flattened_list = [result 
                                        for result in taskResult 
                                        if result is not None]

                    logger.info("Receiving %d results", len(flattened_list))

                    # Append to matrix
                    matrix = matrix + flattened_list

            self._appendListToFile(os.path.join( self.outputFolder, matrix_filename ) + ".csv", matrix )
        
            threadPool.submit( self._compressFiles, 
                            isochrones_dir, 
                            '*.geojson',
                            f'runDate_{runDate}_modelDate_{travel_datetime.isoformat()}_batch{idx+1:03}.geoJson.zip' )                

        logger.info("compressing travel time matrix file (%s)", matrix_filename )        

        self._compressFiles( self.outputFolder,
                                    matrix_filename + ".csv",
                                    matrix_filename + ".zip")
        self._csv_header_written = False

        return errorCount

                          



class _IsochroneJobExecutorGroupByLocation(_IsochroneJobExecutorBase):
    def __init__(self, outputFolder :str, config :ProcessConfig):
        super().__init__(outputFolder, config)

    def _jobSorterFn(self,item):
        return (item["name"], item["mode"], item["travel_time"])

    def _jobGrouperFn(self,item):
        return (item["name"])
    
    def _jobGrouperInnerFn(self,item):
        return (item["name"], item["mode"])

    def _executeJobs(    self,
                            threadPool :concurrent.futures.ThreadPoolExecutor,
                            isochrones_dir :str,
                            jobs :list) -> int:           
        errorCount = 0
        
        runDate = datetime.date.today().isoformat()

        matrix_filename = f"AreatoAreaTravelTimeMatrix_ToArriveBy_{self.config.date.strftime('%Y%m%d')}"
     
        with open(os.path.join( self.outputFolder, matrix_filename ) + ".csv", 'w') as file:
            pass  # truncate the file and close it again
       
        self._csv_header_written = False
        
        groupedJobs = [list(group) for _, group in itertools.groupby(jobs, key=self._jobGrouperFn)]

        numChunks = math.ceil(len(groupedJobs) / (self.config.number_of_threads * 5))
            
        for idx, batch in enumerate(chunker(groupedJobs, self.config.number_of_threads*5)):
            logger.info(
                "==================== Running batch %d of %d ====================", idx+1, numChunks)
            logger.info("Dispatching %d zones", len(batch))

            errorCount += self._runJobGroup(threadPool, 
                                            isochrones_dir, 
                                            batch, 
                                            os.path.join( self.outputFolder, matrix_filename ), 
                                            runDate) 

            threadPool.submit( self._compressFiles, 
                               isochrones_dir, 
                               '*.geojson',
                              f'runDate_{runDate}_modelDate_{self.config.date.isoformat()}_batch{idx+1:03}.geoJson.zip' )
        
        #wait for CSV writes to complete
        threadPool.shutdown()

        logger.info("compressing travel time matrix file (%s)", matrix_filename)        
        
        self._compressFiles( self.outputFolder,
                                    matrix_filename + ".csv",
                                    matrix_filename + ".zip")
        self._csv_header_written = False

        return errorCount


    def _runJobGroup(self, 
                            threadPool :concurrent.futures.ThreadPoolExecutor,
                            isochrones_dir :str,
                            jobList :list,
                            matrix_filename :str,
                            runDate :str) -> int:   
        errorCount = 0

        flattened_list = [item for innerList in jobList for item in innerList]

        logger.info("Dispatching %d jobs", len(flattened_list))
        
        taskFutures = [threadPool.submit(run_batch_catch_errors, list(job)) 
                        for _,job in itertools.groupby(flattened_list, key=self._jobGrouperInnerFn)] 

        matrix = list()
      
        for task in concurrent.futures.as_completed(taskFutures):
            taskResult = task.result()

            if taskResult is None:  
                logger.error("null taskResult object recieved")
            elif "error" in taskResult:
                errorCount += 1 #message logged in the execution thread
            else:
                # This is a list comprehension which flattens the results
                flattened_list = [result 
                                    for result in taskResult 
                                    if result is not None]

                logger.info("Receiving %d results", len(flattened_list))

                # Append to matrix
                matrix = matrix + flattened_list

        threadPool.submit( self._appendListToFile, matrix_filename + ".csv", matrix )                         
                
        return errorCount





def main():   
    arguments = ProcessArgs.parse()

    _process_timer = Timer()
    
    configure_app_logging(logging.INFO, dir=arguments.folder / "logs")

    @atexit.register
    def report_time():
        logger.info("Finished in %s", _process_timer)

    logger.info("Using Python Version: %s", sys.version)

    config = load_config(arguments.folder)

    server = Server(arguments.folder, hostname=config.hostname, port=config.port)

    if arguments.save_parameters: #TODO I have broken this (may not have been working...)
        logger.info("Saving OTP request parameters without starting OTP")
    elif not config.no_server:
        logger.info("Starting server")
        server.start()
  
    jobExector = IsochroneJobExecutor.factory( arguments.folder, config ) 

    errorCount = jobExector.run(server)    

    # Stop OTP Server
    server.stop()    

    if errorCount > 0:
        logger.info("all tasks completed.")
        logger.critical( "ALERT: %d errors occurred during run. Check log file for details.", errorCount )
    else:
        logger.info("All tasks completed without error.")


if __name__ == "__main__":
    main()
