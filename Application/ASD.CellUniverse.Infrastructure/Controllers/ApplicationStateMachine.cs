using System;
using System.Windows.Input;

namespace ASD.CellUniverse.Infrastructure.Controllers {

    using MVVM;
    using Interfaces;

    public sealed class ApplicationStateMachine : BindableBase, IMainController {

        private enum State { Started, Paused, Stopped }
        private object shared = new object();

        private State state = State.Stopped;

        private DelegateCommand playCommand, pauseCommand, stopCommand;

        public ICommand Play => playCommand;
        public ICommand Pause => pauseCommand;
        public ICommand Stop => stopCommand;

        internal event Action Started, Paused, Stopped;

        public ApplicationStateMachine() => InitializeCommands();

        private void InitializeCommands() {

            playCommand = new DelegateCommand(
                (o) => ChangeState(State.Started, Started),
                (o) => state == State.Stopped || state == State.Paused);

            pauseCommand = new DelegateCommand(
                (o) => ChangeState(State.Paused, Paused),
                (o) => state == State.Started);

            stopCommand = new DelegateCommand(
                (o) => ChangeState(State.Stopped, Stopped),
                (o) => state == State.Started);
        }

        private void ChangeState(State newState, Action onNewState) {
            lock (shared) {
                state = newState;
                onNewState?.Invoke();
            }
        }
    }
}