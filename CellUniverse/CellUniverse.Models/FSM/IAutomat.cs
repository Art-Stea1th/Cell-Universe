namespace CellUniverse.Models.FSM {


    using States;

    internal interface IAutomat {

        IState Started { get; }
        IState Halted { get; }
        IState Terminated { get; }

        void SetState(IState newState);

        void Start();
        void Halt();
        void Terminate();        
    }
}