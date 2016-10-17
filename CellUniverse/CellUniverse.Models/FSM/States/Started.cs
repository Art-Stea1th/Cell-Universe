namespace CellUniverse.Models.FSM.States {


    class Started : IState {

        private IAutomat automat;

        public Started(IAutomat automat) {
            this.automat = automat;
        }

        void IState.Start() { }

        void IState.Halt() {
            automat.SetState(automat.Halted);
        }

        void IState.Terminate() {
            automat.SetState(automat.Terminated);
        }
    }
}