namespace CellUniverse.Models.FSM.States {


    class Halted : IState {

        private IAutomat automat;

        public Halted(IAutomat automat) {
            this.automat = automat;
        }

        void IState.Start() {
            automat.SetState(automat.Started);
        }

        void IState.Halt() { }

        void IState.Terminate() {
            automat.SetState(automat.Terminated);
        }
    }
}