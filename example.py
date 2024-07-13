from customizable_synthesizer import CuTS


program = '''
SYNTHESIZE: Adult;

    ENFORCE: STATISTICAL:  
        E[age|age > 30] == 40;
    
END;
'''    

cuts = CuTS(program)
cuts.fit(verbose=True)

syndata = cuts.generate_data(30000)
