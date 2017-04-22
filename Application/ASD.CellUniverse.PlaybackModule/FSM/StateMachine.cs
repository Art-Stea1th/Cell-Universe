using System;
using System.Windows.Input;
using Prism.Commands;

namespace ASD.CellUniverse.PlaybackModule.FSM {

    internal sealed class StateMachine {

        private enum State { Started, Paused, Stopped }
        private object shared = new object();

        private State state = State.Stopped;

        private DelegateCommand playCommand, pauseCommand, stopCommand;

        public ICommand Play => playCommand;
        public ICommand Pause => pauseCommand;
        public ICommand Stop => stopCommand;

        public event Action Started, Paused, Stopped;

        public StateMachine() => InitializeCommands();

        private void InitializeCommands() {

            playCommand = new DelegateCommand(
                () => ChangeState(State.Started, Started),
                () => state == State.Stopped || state == State.Paused);

            pauseCommand = new DelegateCommand(
                () => ChangeState(State.Paused, Paused),
                () => state == State.Started);

            stopCommand = new DelegateCommand(
                () => ChangeState(State.Stopped, Stopped),
                () => state == State.Started);
        }

        private void ChangeState(State newState, Action onNewState) {
            lock (shared) {
                state = newState;
                onNewState?.Invoke();
                Raise();
            }
        }

        private void Raise() {
            playCommand.RaiseCanExecuteChanged();
            pauseCommand.RaiseCanExecuteChanged();
            stopCommand.RaiseCanExecuteChanged();
        }
    }
}