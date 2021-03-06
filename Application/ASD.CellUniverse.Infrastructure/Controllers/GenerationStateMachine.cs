﻿using System;
using System.Windows.Input;

namespace ASD.CellUniverse.Infrastructure.Controllers {

    using MVVM;
    using Interfaces;

    internal sealed class GenerationStateMachine : BindableBase, IGenerationController {

        private object shared = new object();
        private State state;

        private StateMachineCommand playCommand, pauseCommand, resumeCommand, stopCommand, resetCommand, dummyCommand;
        private StateMachineCommand playPauseResumeCommand, stopResetCommand;

        public event Action Started, Paused, Resumed, Stopped, Reseted;
        public event Action<State> StateChanged;

        public State State {
            get => state;
            private set => Set(ref state, value);
        }

        public ICommand Start {
            get => playPauseResumeCommand;
            private set => Set(ref playPauseResumeCommand, value as StateMachineCommand);
        }

        public ICommand Stop {
            get => stopResetCommand;
            private set => Set(ref stopResetCommand, value as StateMachineCommand);
        }

        internal GenerationStateMachine() => InitializeCommands();

        private void InitializeCommands() {

            playCommand = new StateMachineCommand(
                "START",
                (o) => ChangeState(State.Started, pauseCommand, stopCommand, Started),
                (o) => State == State.Stopped);

            pauseCommand = new StateMachineCommand(
                "PAUSE",
                (o) => ChangeState(State.Paused, resumeCommand, resetCommand, Paused),
                (o) => State == State.Started);

            resumeCommand = new StateMachineCommand(
                "RESUME",
                (o) => ChangeState(State.Started, pauseCommand, stopCommand, Resumed),
                (o) => State == State.Paused);


            stopCommand = new StateMachineCommand(
                "STOP",
                (o) => ChangeState(State.Stopped, playCommand, dummyCommand, Stopped),
                (o) => State == State.Started);

            resetCommand = new StateMachineCommand(
                "RESET",
                (o) => ChangeState(State.Stopped, playCommand, dummyCommand, Reseted),
                (o) => State == State.Paused);

            dummyCommand = new StateMachineCommand(
                "...",
                (o) => { },
                (o) => 0.0 == double.Epsilon);

            ChangeState(State.Stopped, playCommand, dummyCommand, null);
        }

        private void ChangeState(
            State newState, StateMachineCommand newPlayCommand, StateMachineCommand newStopCommand, Action onNewState) {

            lock (shared) {
                State = newState;

                Start = newPlayCommand;
                Stop = newStopCommand;

                onNewState?.Invoke();
                StateChanged?.Invoke(State);
            }
        }

        private class StateMachineCommand : RelayCommand {

            internal string Name { get; }

            public StateMachineCommand(string name, Action<object> execute, Predicate<object> canExecute)
                : base(execute, canExecute) {
                Name = name;
            }
            public override string ToString() => Name;
        }
    }
}