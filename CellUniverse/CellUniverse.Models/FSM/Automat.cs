namespace CellUniverse.Models.FSM {


    using States;

    internal class Automat : IAutomat {

        private IState started;
        private IState halted;
        private IState terminated;

        private IState currentState;

        IState IAutomat.Started    { get { return started; } }
        IState IAutomat.Halted     { get { return halted; } }
        IState IAutomat.Terminated { get { return terminated; } }

        internal Automat() {
            started = new Started(this);
            halted = new Halted(this);
            terminated = new Terminated(this);
        }

        void IAutomat.SetState(IState newState) {
            currentState = newState;
        }

        void IAutomat.Start() {
            currentState.Start();
        }

        void IAutomat.Halt() {
            currentState.Halt();
        }

        void IAutomat.Terminate() {
            currentState.Terminate();
        }
    }
}