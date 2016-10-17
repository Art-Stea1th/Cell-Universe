namespace CellUniverse.Models.FSM.States {


    class Terminated : IState {

        private IAutomat automat;

        public Terminated(IAutomat automat) {
            this.automat = automat;
        }

        void IState.Start() {
            automat.SetState(automat.Started);
        }

        void IState.Halt() { }
        void IState.Terminate() { }
    }
}