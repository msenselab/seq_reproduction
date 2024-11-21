#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Thu Nov 21 16:32:13 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2023.2.3')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_5
fb_text  = ''
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'seq_tasks_v2023'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):04.0f}",
    'gender (M/F)': '',
    'age': '',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s' % (expName, expInfo['participant'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/jiaowu/mlabRepo/seq_follow/seq_follow_pilot/seq_tasks_v2023_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.ERROR)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.ERROR)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1280, 1024], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='ptb')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='PsychToolbox')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instruction" ---
    text_inst = visual.TextStim(win=win, name='text_inst',
        text="Experimental Instruction\n\nWelcome to our reproduction experiment. In this experiment, you will see a Gabor patch appears on the screen. You need to remember the DURATION of the Gabor patch. After it disappears, a cue (either 'R' or 'F') will appear. \n\nWhen the cue is 'R', you are asked to press the 'Down' Arrow key as long as what you perceived. The key press will show a Gabor patch again, helping you compare the last duration. \n\nIf the cue is 'F', a Gabor patch will automatically appear. Your task is to press the 'Down' arrow key as soon as it appears and release the key when it disappears. \n\n\nThe whole experiment consists of a 1-block practice and a 12-block formal task.\n\nPress SPACE to start Practice...\n",
        font='Arial',
        pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_key = keyboard.Keyboard()
    
    # --- Initialize components for Routine "blockPrac" ---
    Prac_info = visual.TextStim(win=win, name='Prac_info',
        text='Practice\n\n\n\nPress SPACE key to start ...',
        font='Arial',
        pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_start = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Encoding" ---
    stimulus_grating = visual.GratingStim(
        win=win, name='stimulus_grating',
        tex='sin', mask='gauss', anchor='center',
        ori=1.0, pos=(0, 0), size=(5, 5), sf=1.0, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=0.25, blendmode='avg',
        texRes=128.0, interpolate=True, depth=0.0)
    # Run 'Begin Experiment' code from code_encoding
    # generate random orientation
    
    orientation = randint(1,180)
    
    durations = np.arange(0.8, 1.5, 0.1)
    fixation = visual.ShapeStim(
        win=win, name='fixation', vertices='cross',
        size=(0.5, 0.5),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "retro_cue" ---
    # Run 'Begin Experiment' code from code_retro_cue
    cue_text = ''
    cue_word = visual.TextStim(win=win, name='cue_word',
        text='',
        font='Arial',
        pos=(0, 0), height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Reproduction" ---
    reproduced_grating = visual.GratingStim(
        win=win, name='reproduced_grating',
        tex='sin', mask='gauss', anchor='center',
        ori=1.0, pos=(0, 0), size=(5, 5), sf=1.0, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=0.25, blendmode='avg',
        texRes=128.0, interpolate=True, depth=0.0)
    # Run 'Begin Experiment' code from code_reproduction
    kb = keyboard.Keyboard()
    
    repDuration = 0
    
    cue_rts = [] # store the cue to reproductin rt
    
    
    # --- Initialize components for Routine "Follow" ---
    # Run 'Begin Experiment' code from code_follow
    import random
    
    followRsp = False
    
    gap = 1 # cue onset to the following stimulus onset
    follow_grating = visual.GratingStim(
        win=win, name='follow_grating',
        tex='sin', mask='gauss', anchor='center',
        ori=1.0, pos=(0, 0), size=(5, 5), sf=1.0, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=0.25, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-1.0)
    filler_blank = visual.TextStim(win=win, name='filler_blank',
        text=None,
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "feedback" ---
    feedback_text = visual.TextStim(win=win, name='feedback_text',
        text='',
        font='Arial',
        pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "iti" ---
    iti_blank = visual.TextStim(win=win, name='iti_blank',
        text=None,
        font='Arial',
        pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "BlockInfo" ---
    text_rest = visual.TextStim(win=win, name='text_rest',
        text='You did a great job! Please take a rest! \n\n\n\n\n\n\n\n\nPress SPACE key to continue ...',
        font='Arial',
        pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    blk_infotext = visual.TextStim(win=win, name='blk_infotext',
        text='',
        font='Arial',
        pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    block_key = keyboard.Keyboard()
    # Run 'Begin Experiment' code from code_blockinfo
    blk_info = ''
    
    # --- Initialize components for Routine "Goodbye" ---
    goodbye_text = visual.TextStim(win=win, name='goodbye_text',
        text='The whole experiment is completed! \n\nMany thanks for your participation!',
        font='Arial',
        pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    goodbye_key = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    instr_key.keys = []
    instr_key.rt = []
    _instr_key_allKeys = []
    # keep track of which components have finished
    InstructionComponents = [text_inst, instr_key]
    for thisComponent in InstructionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_inst* updates
        
        # if text_inst is starting this frame...
        if text_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_inst.frameNStart = frameN  # exact frame index
            text_inst.tStart = t  # local t and not account for scr refresh
            text_inst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_inst, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_inst.status = STARTED
            text_inst.setAutoDraw(True)
        
        # if text_inst is active this frame...
        if text_inst.status == STARTED:
            # update params
            pass
        
        # *instr_key* updates
        waitOnFlip = False
        
        # if instr_key is starting this frame...
        if instr_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key.frameNStart = frameN  # exact frame index
            instr_key.tStart = t  # local t and not account for scr refresh
            instr_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key, 'tStartRefresh')  # time at next scr refresh
            # update status
            instr_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key.status == STARTED and not waitOnFlip:
            theseKeys = instr_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_allKeys.extend(theseKeys)
            if len(_instr_key_allKeys):
                instr_key.keys = _instr_key_allKeys[-1].name  # just the last key pressed
                instr_key.rt = _instr_key_allKeys[-1].rt
                instr_key.duration = _instr_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in InstructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruction" ---
    for thisComponent in InstructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "Instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "blockPrac" ---
    continueRoutine = True
    # update component parameters for each repeat
    prac_start.keys = []
    prac_start.rt = []
    _prac_start_allKeys = []
    # keep track of which components have finished
    blockPracComponents = [Prac_info, prac_start]
    for thisComponent in blockPracComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "blockPrac" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Prac_info* updates
        
        # if Prac_info is starting this frame...
        if Prac_info.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Prac_info.frameNStart = frameN  # exact frame index
            Prac_info.tStart = t  # local t and not account for scr refresh
            Prac_info.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Prac_info, 'tStartRefresh')  # time at next scr refresh
            # update status
            Prac_info.status = STARTED
            Prac_info.setAutoDraw(True)
        
        # if Prac_info is active this frame...
        if Prac_info.status == STARTED:
            # update params
            pass
        
        # *prac_start* updates
        waitOnFlip = False
        
        # if prac_start is starting this frame...
        if prac_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_start.frameNStart = frameN  # exact frame index
            prac_start.tStart = t  # local t and not account for scr refresh
            prac_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            prac_start.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(prac_start.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(prac_start.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if prac_start.status == STARTED and not waitOnFlip:
            theseKeys = prac_start.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _prac_start_allKeys.extend(theseKeys)
            if len(_prac_start_allKeys):
                prac_start.keys = _prac_start_allKeys[-1].name  # just the last key pressed
                prac_start.rt = _prac_start_allKeys[-1].rt
                prac_start.duration = _prac_start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in blockPracComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blockPrac" ---
    for thisComponent in blockPracComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "blockPrac" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=13.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions.xlsx'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Encoding" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_encoding
        orientation = randint(1,180)
        
        # keep track of which components have finished
        EncodingComponents = [stimulus_grating, fixation]
        for thisComponent in EncodingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Encoding" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *stimulus_grating* updates
            
            # if stimulus_grating is starting this frame...
            if stimulus_grating.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                stimulus_grating.frameNStart = frameN  # exact frame index
                stimulus_grating.tStart = t  # local t and not account for scr refresh
                stimulus_grating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stimulus_grating, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stimulus_grating.started')
                # update status
                stimulus_grating.status = STARTED
                stimulus_grating.setAutoDraw(True)
            
            # if stimulus_grating is active this frame...
            if stimulus_grating.status == STARTED:
                # update params
                stimulus_grating.setOri(orientation, log=False)
            
            # if stimulus_grating is stopping this frame...
            if stimulus_grating.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stimulus_grating.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    stimulus_grating.tStop = t  # not accounting for scr refresh
                    stimulus_grating.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stimulus_grating.stopped')
                    # update status
                    stimulus_grating.status = FINISHED
                    stimulus_grating.setAutoDraw(False)
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in EncodingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Encoding" ---
        for thisComponent in EncodingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "Encoding" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "retro_cue" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_retro_cue
        if cue == "reproduction":
            cue_text = 'R'
        else:
            cue_text = 'F'
        
        cue_onset = globalClock.getTime()
        cue_word.setText(cue_text)
        # keep track of which components have finished
        retro_cueComponents = [cue_word]
        for thisComponent in retro_cueComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "retro_cue" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cue_word* updates
            
            # if cue_word is starting this frame...
            if cue_word.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
                # keep track of start time/frame for later
                cue_word.frameNStart = frameN  # exact frame index
                cue_word.tStart = t  # local t and not account for scr refresh
                cue_word.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_word, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_word.started')
                # update status
                cue_word.status = STARTED
                cue_word.setAutoDraw(True)
            
            # if cue_word is active this frame...
            if cue_word.status == STARTED:
                # update params
                pass
            
            # if cue_word is stopping this frame...
            if cue_word.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_word.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_word.tStop = t  # not accounting for scr refresh
                    cue_word.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_word.stopped')
                    # update status
                    cue_word.status = FINISHED
                    cue_word.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in retro_cueComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "retro_cue" ---
        for thisComponent in retro_cueComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "Reproduction" ---
        continueRoutine = True
        # update component parameters for each repeat
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (cue == 'following')
        # Run 'Begin Routine' code from code_reproduction
        kb.clearEvents(eventType='keyboard')
        event.clearEvents(eventType='keyboard')
        
        key = kb.getKeys(['down'], waitRelease = False, clear = False)
        
        key_pressed = False
        key_released = True
        
        key_onset = 0
        key_offset = 0
        
        repDuration = 0
        # keep track of which components have finished
        ReproductionComponents = [reproduced_grating]
        for thisComponent in ReproductionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Reproduction" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *reproduced_grating* updates
            
            # if reproduced_grating is starting this frame...
            if reproduced_grating.status == NOT_STARTED and key_pressed:
                # keep track of start time/frame for later
                reproduced_grating.frameNStart = frameN  # exact frame index
                reproduced_grating.tStart = t  # local t and not account for scr refresh
                reproduced_grating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reproduced_grating, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reproduced_grating.started')
                # update status
                reproduced_grating.status = STARTED
                reproduced_grating.setAutoDraw(True)
            
            # if reproduced_grating is active this frame...
            if reproduced_grating.status == STARTED:
                # update params
                reproduced_grating.setOri(orientation, log=False)
            
            # if reproduced_grating is stopping this frame...
            if reproduced_grating.status == STARTED:
                if bool(key_released):
                    # keep track of stop time/frame for later
                    reproduced_grating.tStop = t  # not accounting for scr refresh
                    reproduced_grating.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reproduced_grating.stopped')
                    # update status
                    reproduced_grating.status = FINISHED
                    reproduced_grating.setAutoDraw(False)
            # Run 'Each Frame' code from code_reproduction
            keys = kb.getKeys(['down'], waitRelease=False,clear = False)
            
            for key in keys:
                if key.duration is None:
                    key_pressed = True
                    key_released = False
                    if key_onset == 0:
                        key_onset = globalClock.getTime()
            
            if key_pressed: # after key pressed
                if isinstance(keys[0].duration, float):
                    key_released = True
                    key_offset = globalClock.getTime()#keys[0].duration
                    repDuration = key_offset - key_onset
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ReproductionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Reproduction" ---
        for thisComponent in ReproductionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from code_reproduction
        kb.clearEvents(eventType='keyboard')
        event.clearEvents(eventType='keyboard')
        
        #store random orientation in the data
        thisExp.addData('orientation', orientation)
        thisExp.addData('rpr_onset', key_onset)
        thisExp.addData('rpr_duration', repDuration)
        
        if cue == "reproduction":
            cue2rep_rt = key_onset - cue_onset
            cue_rts.append(cue2rep_rt)
        else:
            cue2rep_rt = 0
        thisExp.addData('cue2rep_rt', cue2rep_rt)
        # the Routine "Reproduction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Follow" ---
        continueRoutine = True
        # update component parameters for each repeat
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (cue == 'reproduction')
        # Run 'Begin Routine' code from code_follow
        kb.clearEvents(eventType='keyboard')
        event.clearEvents(eventType='keyboard')
        
        key = kb.getKeys(['down'], waitRelease = False, clear = False)
        
        key_pressed = False
        key_released = True
        
        key_follow_onset = 0
        key_follow_offset = 0
        
        followProduction = 0
        followRsp = False
        
        follow_dur = random.choice(durations)
        # keep track of which components have finished
        FollowComponents = [follow_grating, filler_blank]
        for thisComponent in FollowComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Follow" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_follow
            keys = kb.getKeys(['down'], waitRelease=False,clear = False)
            
            for key in keys:
                if key.duration is None:
                    key_pressed = True
                    key_released = False
                    if key_follow_onset == 0:
                        key_follow_onset = globalClock.getTime()
            
            if key_pressed: # after key pressed
                if isinstance(keys[0].duration, float):
                    key_released = True
                    key_follow_offset = globalClock.getTime()
                    followProduction = key_follow_offset - key_follow_onset
                    followRsp = True
            
            # *follow_grating* updates
            
            # if follow_grating is starting this frame...
            if follow_grating.status == NOT_STARTED and tThisFlip >= gap-frameTolerance:
                # keep track of start time/frame for later
                follow_grating.frameNStart = frameN  # exact frame index
                follow_grating.tStart = t  # local t and not account for scr refresh
                follow_grating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(follow_grating, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'follow_grating.started')
                # update status
                follow_grating.status = STARTED
                follow_grating.setAutoDraw(True)
            
            # if follow_grating is active this frame...
            if follow_grating.status == STARTED:
                # update params
                follow_grating.setOri(orientation, log=False)
            
            # if follow_grating is stopping this frame...
            if follow_grating.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > follow_grating.tStartRefresh + follow_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    follow_grating.tStop = t  # not accounting for scr refresh
                    follow_grating.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'follow_grating.stopped')
                    # update status
                    follow_grating.status = FINISHED
                    follow_grating.setAutoDraw(False)
            
            # *filler_blank* updates
            
            # if filler_blank is starting this frame...
            if filler_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                filler_blank.frameNStart = frameN  # exact frame index
                filler_blank.tStart = t  # local t and not account for scr refresh
                filler_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(filler_blank, 'tStartRefresh')  # time at next scr refresh
                # update status
                filler_blank.status = STARTED
                filler_blank.setAutoDraw(True)
            
            # if filler_blank is active this frame...
            if filler_blank.status == STARTED:
                # update params
                pass
            
            # if filler_blank is stopping this frame...
            if filler_blank.status == STARTED:
                if bool(followRsp):
                    # keep track of stop time/frame for later
                    filler_blank.tStop = t  # not accounting for scr refresh
                    filler_blank.frameNStop = frameN  # exact frame index
                    # update status
                    filler_blank.status = FINISHED
                    filler_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FollowComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Follow" ---
        for thisComponent in FollowComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from code_follow
        kb.clearEvents(eventType='keyboard')
        event.clearEvents(eventType='keyboard')
        
        #store random orientation in the data
        thisExp.addData('follow_stimuli', follow_dur)
        thisExp.addData('follow_key_onset',key_follow_onset)
        thisExp.addData('follow_production', followProduction)
        # the Routine "Follow" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('feedback.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (trials.thisN >= 28)
        # Run 'Begin Routine' code from code_5
        error_ratio = np.min([np.abs(repDuration/duration -1), np.abs(followProduction/follow_dur -1)])
        
        if error_ratio > 0.3:
            fb_text = "Your key pressed duration deviated too far!"
        else:
            fb_text = ''
        # keep track of which components have finished
        feedbackComponents = [feedback_text]
        for thisComponent in feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.8:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_text* updates
            
            # if feedback_text is starting this frame...
            if feedback_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                feedback_text.frameNStart = frameN  # exact frame index
                feedback_text.tStart = t  # local t and not account for scr refresh
                feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
                # update status
                feedback_text.status = STARTED
                feedback_text.setAutoDraw(True)
            
            # if feedback_text is active this frame...
            if feedback_text.status == STARTED:
                # update params
                feedback_text.setText(fb_text, log=False)
            
            # if feedback_text is stopping this frame...
            if feedback_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_text.tStartRefresh + 0.8-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_text.tStop = t  # not accounting for scr refresh
                    feedback_text.frameNStop = frameN  # exact frame index
                    # update status
                    feedback_text.status = FINISHED
                    feedback_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('feedback.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.800000)
        
        # --- Prepare to start Routine "iti" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('iti.started', globalClock.getTime())
        # keep track of which components have finished
        itiComponents = [iti_blank]
        for thisComponent in itiComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *iti_blank* updates
            
            # if iti_blank is starting this frame...
            if iti_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                iti_blank.frameNStart = frameN  # exact frame index
                iti_blank.tStart = t  # local t and not account for scr refresh
                iti_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(iti_blank, 'tStartRefresh')  # time at next scr refresh
                # update status
                iti_blank.status = STARTED
                iti_blank.setAutoDraw(True)
            
            # if iti_blank is active this frame...
            if iti_blank.status == STARTED:
                # update params
                pass
            
            # if iti_blank is stopping this frame...
            if iti_blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > iti_blank.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    iti_blank.tStop = t  # not accounting for scr refresh
                    iti_blank.frameNStop = frameN  # exact frame index
                    # update status
                    iti_blank.status = FINISHED
                    iti_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in itiComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti" ---
        for thisComponent in itiComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('iti.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "BlockInfo" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BlockInfo.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not ((trials.thisN +1) % 56 > 0)
        block_key.keys = []
        block_key.rt = []
        _block_key_allKeys = []
        # Run 'Begin Routine' code from code_blockinfo
        blknum = 1 + trials.thisN // 56
        blk_info = f"Block {blknum}"
        print(blk_info)
        
        if trials.thisN == 55: # last trial of the first practice block
            last14Following = cue_rts[-14:]
            gap = np.mean(last14Following) - 1
        # keep track of which components have finished
        BlockInfoComponents = [text_rest, blk_infotext, block_key]
        for thisComponent in BlockInfoComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BlockInfo" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_rest* updates
            
            # if text_rest is starting this frame...
            if text_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_rest.frameNStart = frameN  # exact frame index
                text_rest.tStart = t  # local t and not account for scr refresh
                text_rest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_rest, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_rest.status = STARTED
                text_rest.setAutoDraw(True)
            
            # if text_rest is active this frame...
            if text_rest.status == STARTED:
                # update params
                pass
            
            # *blk_infotext* updates
            
            # if blk_infotext is starting this frame...
            if blk_infotext.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blk_infotext.frameNStart = frameN  # exact frame index
                blk_infotext.tStart = t  # local t and not account for scr refresh
                blk_infotext.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blk_infotext, 'tStartRefresh')  # time at next scr refresh
                # update status
                blk_infotext.status = STARTED
                blk_infotext.setAutoDraw(True)
            
            # if blk_infotext is active this frame...
            if blk_infotext.status == STARTED:
                # update params
                blk_infotext.setText(blk_info, log=False)
            
            # *block_key* updates
            waitOnFlip = False
            
            # if block_key is starting this frame...
            if block_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_key.frameNStart = frameN  # exact frame index
                block_key.tStart = t  # local t and not account for scr refresh
                block_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_key, 'tStartRefresh')  # time at next scr refresh
                # update status
                block_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(block_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(block_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if block_key.status == STARTED and not waitOnFlip:
                theseKeys = block_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _block_key_allKeys.extend(theseKeys)
                if len(_block_key_allKeys):
                    block_key.keys = _block_key_allKeys[-1].name  # just the last key pressed
                    block_key.rt = _block_key_allKeys[-1].rt
                    block_key.duration = _block_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BlockInfoComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BlockInfo" ---
        for thisComponent in BlockInfoComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BlockInfo.stopped', globalClock.getTime())
        # the Routine "BlockInfo" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 13.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "Goodbye" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Goodbye.started', globalClock.getTime())
    goodbye_key.keys = []
    goodbye_key.rt = []
    _goodbye_key_allKeys = []
    # keep track of which components have finished
    GoodbyeComponents = [goodbye_text, goodbye_key]
    for thisComponent in GoodbyeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Goodbye" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *goodbye_text* updates
        
        # if goodbye_text is starting this frame...
        if goodbye_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            goodbye_text.frameNStart = frameN  # exact frame index
            goodbye_text.tStart = t  # local t and not account for scr refresh
            goodbye_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(goodbye_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'goodbye_text.started')
            # update status
            goodbye_text.status = STARTED
            goodbye_text.setAutoDraw(True)
        
        # if goodbye_text is active this frame...
        if goodbye_text.status == STARTED:
            # update params
            pass
        
        # *goodbye_key* updates
        waitOnFlip = False
        
        # if goodbye_key is starting this frame...
        if goodbye_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            goodbye_key.frameNStart = frameN  # exact frame index
            goodbye_key.tStart = t  # local t and not account for scr refresh
            goodbye_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(goodbye_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'goodbye_key.started')
            # update status
            goodbye_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(goodbye_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(goodbye_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if goodbye_key.status == STARTED and not waitOnFlip:
            theseKeys = goodbye_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _goodbye_key_allKeys.extend(theseKeys)
            if len(_goodbye_key_allKeys):
                goodbye_key.keys = _goodbye_key_allKeys[-1].name  # just the last key pressed
                goodbye_key.rt = _goodbye_key_allKeys[-1].rt
                goodbye_key.duration = _goodbye_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in GoodbyeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Goodbye" ---
    for thisComponent in GoodbyeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Goodbye.stopped', globalClock.getTime())
    # check responses
    if goodbye_key.keys in ['', [], None]:  # No response was made
        goodbye_key.keys = None
    thisExp.addData('goodbye_key.keys',goodbye_key.keys)
    if goodbye_key.keys != None:  # we had a response
        thisExp.addData('goodbye_key.rt', goodbye_key.rt)
        thisExp.addData('goodbye_key.duration', goodbye_key.duration)
    thisExp.nextEntry()
    # the Routine "Goodbye" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
