namespace CellUniverse.Models.FSM.States {


    public interface IState {

        void Start();
        void Halt();
        void Terminate();
    }
}