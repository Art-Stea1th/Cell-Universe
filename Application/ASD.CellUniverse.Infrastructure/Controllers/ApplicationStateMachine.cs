using System;
using System.Windows.Input;

namespace ASD.CellUniverse.Infrastructure.Controllers {

    using MVVM;
    using Interfaces;

    public sealed class ApplicationStateMachine : BindableBase, IMainController {        

        private enum State { Started, Paused, Stopped }
        private object shared = new object();

        private State state;

        private StateMachineCommand playCommand, pauseCommand, resumeCommand, stopCommand, resetCommand, dummyCommand;

        private StateMachineCommand playPauseResumeCommand, stopResetCommand;
        private string playPauseResumeName, stopResetName;

        public event Action Started, Paused, Resumed, Stopped, Reseted;

        public ICommand PlayPauseResume {
            get => playPauseResumeCommand;
            private set => SetProperty(ref playPauseResumeCommand, value as StateMachineCommand);
        }

        public ICommand StopReset {
            get => stopResetCommand;
            private set => SetProperty(ref stopResetCommand, value as StateMachineCommand);
        }

        public string PlayPauseResumeName {
            get => playPauseResumeName;
            set => SetProperty(ref playPauseResumeName, value);
        }

        public string StopResetName {
            get => stopResetName;
            set => SetProperty(ref stopResetName, value);
        }

        public ApplicationStateMachine() => InitializeCommands();

        private void InitializeCommands() {

            playCommand = new StateMachineCommand(
                "PLAY",
                (o) => ChangeState(State.Started, pauseCommand, stopCommand, Started),
                (o) => state == State.Stopped);

            pauseCommand = new StateMachineCommand(
                "PAUSE",
                (o) => ChangeState(State.Paused, resumeCommand, resetCommand, Paused),
                (o) => state == State.Started);

            resumeCommand = new StateMachineCommand(
                "RESUME",
                (o) => ChangeState(State.Started, pauseCommand, stopCommand, Resumed),
                (o) => state == State.Paused);


            stopCommand = new StateMachineCommand(
                "STOP",
                (o) => ChangeState(State.Stopped, playCommand, dummyCommand, Stopped),
                (o) => state == State.Started);

            resetCommand = new StateMachineCommand(
                "RESET",
                (o) => ChangeState(State.Stopped, playCommand, dummyCommand, Reseted),
                (o) => state == State.Paused);

            dummyCommand = new StateMachineCommand(
                "...",
                (o) => { },
                (o) => 0 == 1);

            ChangeState(State.Stopped, playCommand, dummyCommand, null);
        }

        private void ChangeState(
            State newState, StateMachineCommand newPlayCommand, StateMachineCommand newStopCommand, Action onNewState) {

            lock (shared) {
                state = newState;

                PlayPauseResume = newPlayCommand;
                StopReset = newStopCommand;

                PlayPauseResumeName = newPlayCommand.Name;
                StopResetName = newStopCommand.Name;

                onNewState?.Invoke();
            }
        }

        private class StateMachineCommand : RelayCommand {

            internal string Name { get; }

            public StateMachineCommand(string name, Action<object> execute, Predicate<object> canExecute)
                : base(execute, canExecute) {
                Name = name;
            }
        }
    }
}